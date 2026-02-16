"""
Reporting Agent for Road Anomaly Detection
Handles database logging and automated reporting to authorities
"""

import sqlite3
import json
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime, timedelta
from pathlib import Path
import threading
import queue
import time
import cv2
import numpy as np


class DetectionDatabase:
    """
    SQLite database for storing detections
    """
    
    def __init__(self, db_path='detections.db'):
        """
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.conn = None
        self.cursor = None
        
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize database and create tables"""
        
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # Create detections table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                class_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                bbox_x1 INTEGER,
                bbox_y1 INTEGER,
                bbox_x2 INTEGER,
                bbox_y2 INTEGER,
                latitude REAL,
                longitude REAL,
                altitude REAL,
                speed_kmh REAL,
                heading REAL,
                gps_fix INTEGER,
                image_path TEXT,
                reported INTEGER DEFAULT 0,
                report_id TEXT,
                notes TEXT
            )
        ''')
        
        # Create reports table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                num_detections INTEGER,
                severity_level TEXT,
                latitude REAL,
                longitude REAL,
                status TEXT DEFAULT 'pending',
                sent_at TEXT,
                recipient TEXT,
                response TEXT
            )
        ''')
        
        # Create indices for faster queries
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON detections(timestamp)
        ''')
        
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_location 
            ON detections(latitude, longitude)
        ''')
        
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_reported 
            ON detections(reported)
        ''')
        
        self.conn.commit()
        
        print(f"✅ Database initialized: {self.db_path}")
    
    def insert_detection(self, detection):
        """
        Insert a detection into the database
        
        Args:
            detection: Detection dictionary with GPS data
            
        Returns:
            int: Inserted row ID
        """
        gps = detection.get('gps', {})
        bbox = detection.get('bbox', [0, 0, 0, 0])
        
        self.cursor.execute('''
            INSERT INTO detections (
                timestamp, class_id, class_name, confidence,
                bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                latitude, longitude, altitude, speed_kmh, heading, gps_fix,
                image_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            detection.get('timestamp', datetime.now().isoformat()),
            detection['class_id'],
            detection['class_name'],
            detection['confidence'],
            bbox[0], bbox[1], bbox[2], bbox[3],
            gps.get('latitude', 0.0),
            gps.get('longitude', 0.0),
            gps.get('altitude', 0.0),
            gps.get('speed_kmh', 0.0),
            gps.get('heading', 0.0),
            1 if gps.get('fix', False) else 0,
            detection.get('image_path', '')
        ))
        
        self.conn.commit()
        return self.cursor.lastrowid
    
    def get_unreported_detections(self, min_confidence=0.5):
        """
        Get detections that haven't been reported yet
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            list: List of unreported detections
        """
        self.cursor.execute('''
            SELECT * FROM detections
            WHERE reported = 0 AND confidence >= ?
            ORDER BY timestamp DESC
        ''', (min_confidence,))
        
        columns = [desc[0] for desc in self.cursor.description]
        results = []
        
        for row in self.cursor.fetchall():
            results.append(dict(zip(columns, row)))
        
        return results
    
    def mark_as_reported(self, detection_ids, report_id):
        """Mark detections as reported"""
        
        placeholders = ','.join('?' * len(detection_ids))
        self.cursor.execute(f'''
            UPDATE detections
            SET reported = 1, report_id = ?
            WHERE id IN ({placeholders})
        ''', [report_id] + detection_ids)
        
        self.conn.commit()
    
    def create_report(self, report_data):
        """Create a new report entry"""
        
        self.cursor.execute('''
            INSERT INTO reports (
                report_id, timestamp, num_detections, severity_level,
                latitude, longitude, status, sent_at, recipient
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            report_data['report_id'],
            report_data['timestamp'],
            report_data['num_detections'],
            report_data['severity_level'],
            report_data['latitude'],
            report_data['longitude'],
            report_data.get('status', 'pending'),
            report_data.get('sent_at'),
            report_data.get('recipient', '')
        ))
        
        self.conn.commit()
        return self.cursor.lastrowid
    
    def get_statistics(self, time_range_hours=24):
        """Get detection statistics"""
        
        cutoff_time = (datetime.now() - timedelta(hours=time_range_hours)).isoformat()
        
        self.cursor.execute('''
            SELECT 
                class_name,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence,
                MIN(timestamp) as first_seen,
                MAX(timestamp) as last_seen
            FROM detections
            WHERE timestamp >= ?
            GROUP BY class_name
            ORDER BY count DESC
        ''', (cutoff_time,))
        
        columns = [desc[0] for desc in self.cursor.description]
        results = []
        
        for row in self.cursor.fetchall():
            results.append(dict(zip(columns, row)))
        
        return results
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


class ReportingAgent:
    """
    Automated reporting agent for road anomalies
    """
    
    def __init__(self, db_path='detections.db', config=None):
        """
        Args:
            db_path: Path to database
            config: Configuration dictionary
        """
        self.database = DetectionDatabase(db_path)
        
        self.config = config or {
            'min_confidence': 0.5,
            'report_interval': 300,  # 5 minutes
            'batch_size': 10,
            'email_enabled': False,
            'webhook_enabled': False,
        }
        
        # Threading
        self.running = False
        self.thread = None
        self.detection_queue = queue.Queue()
        
        print("✅ Reporting Agent initialized")
    
    def start(self):
        """Start reporting agent"""
        
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._reporting_loop, daemon=True)
        self.thread.start()
        
        print("📤 Reporting Agent: Started")
    
    def stop(self):
        """Stop reporting agent"""
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        
        self.database.close()
        print("📤 Reporting Agent: Stopped")
    
    def queue_detection(self, detection):
        """
        Queue a detection for processing
        
        Args:
            detection: Detection dictionary with GPS data
        """
        self.detection_queue.put(detection)
    
    def _reporting_loop(self):
        """Main reporting loop"""
        
        last_report_time = time.time()
        
        while self.running:
            try:
                # Process queued detections
                while not self.detection_queue.empty():
                    detection = self.detection_queue.get_nowait()
                    self.database.insert_detection(detection)
                
                # Check if it's time to generate report
                if time.time() - last_report_time >= self.config['report_interval']:
                    self._generate_report()
                    last_report_time = time.time()
                
                time.sleep(1)
                
            except Exception as e:
                print(f"⚠️ Reporting loop error: {e}")
                time.sleep(1)
    
    def _generate_report(self):
        """Generate and send report"""
        
        # Get unreported detections
        detections = self.database.get_unreported_detections(
            min_confidence=self.config['min_confidence']
        )
        
        if not detections:
            return
        
        print(f"\n📤 Generating report for {len(detections)} detections...")
        
        # Create report
        report_id = f"REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Group by location (cluster nearby detections)
        clusters = self._cluster_detections(detections)
        
        for cluster in clusters:
            report_data = {
                'report_id': f"{report_id}_{cluster['id']}",
                'timestamp': datetime.now().isoformat(),
                'num_detections': len(cluster['detections']),
                'severity_level': cluster['severity'],
                'latitude': cluster['center_lat'],
                'longitude': cluster['center_lon'],
                'sent_at': datetime.now().isoformat(),
                'recipient': self.config.get('recipient_email', 'municipal@example.com')
            }
            
            # Save report to database
            self.database.create_report(report_data)
            
            # Send notifications
            if self.config.get('email_enabled', False):
                self._send_email_report(report_data, cluster['detections'])
            
            if self.config.get('webhook_enabled', False):
                self._send_webhook_report(report_data, cluster['detections'])
            
            # Mark detections as reported
            detection_ids = [d['id'] for d in cluster['detections']]
            self.database.mark_as_reported(detection_ids, report_data['report_id'])
            
            print(f"✅ Report sent: {report_data['report_id']}")
    
    def _cluster_detections(self, detections, distance_threshold=50):
        """
        Cluster detections by location
        
        Args:
            detections: List of detections
            distance_threshold: Distance threshold in meters
            
        Returns:
            list: List of clusters
        """
        from gps_agent import calculate_distance
        
        clusters = []
        used = set()
        
        for i, det in enumerate(detections):
            if i in used:
                continue
            
            cluster = {
                'id': len(clusters) + 1,
                'detections': [det],
                'center_lat': det['latitude'],
                'center_lon': det['longitude']
            }
            used.add(i)
            
            # Find nearby detections
            for j, other in enumerate(detections):
                if j in used:
                    continue
                
                distance = calculate_distance(
                    det['latitude'], det['longitude'],
                    other['latitude'], other['longitude']
                )
                
                if distance <= distance_threshold:
                    cluster['detections'].append(other)
                    used.add(j)
            
            # Calculate cluster center
            lats = [d['latitude'] for d in cluster['detections']]
            lons = [d['longitude'] for d in cluster['detections']]
            cluster['center_lat'] = sum(lats) / len(lats)
            cluster['center_lon'] = sum(lons) / len(lons)
            
            # Determine severity
            cluster['severity'] = self._calculate_severity(cluster['detections'])
            
            clusters.append(cluster)
        
        return clusters
    
    def _calculate_severity(self, detections):
        """Calculate severity level based on detections"""
        
        # Severity weights
        severity_weights = {
            'Pothole': 3,
            'Alligator Crack': 4,
            'Longitudinal Crack': 2,
            'Other Damage': 2
        }
        
        total_score = sum(
            severity_weights.get(d['class_name'], 1) * d['confidence']
            for d in detections
        )
        
        avg_score = total_score / len(detections)
        
        if avg_score >= 3.5:
            return 'critical'
        elif avg_score >= 2.5:
            return 'high'
        elif avg_score >= 1.5:
            return 'medium'
        else:
            return 'low'
    
    def _send_email_report(self, report_data, detections):
        """Send email report"""
        
        try:
            from gps_agent import generate_maps_link
            
            # Email configuration (from config)
            smtp_server = self.config.get('smtp_server', 'smtp.gmail.com')
            smtp_port = self.config.get('smtp_port', 587)
            sender_email = self.config.get('sender_email', '')
            sender_password = self.config.get('sender_password', '')
            recipient_email = report_data['recipient']
            
            if not sender_email or not sender_password:
                print("⚠️ Email credentials not configured")
                return
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"Road Anomaly Report - {report_data['severity_level'].upper()} Priority"
            msg['From'] = sender_email
            msg['To'] = recipient_email
            
            # Create email body
            maps_link = generate_maps_link(report_data['latitude'], report_data['longitude'])
            
            html_body = f"""
            <html>
              <body>
                <h2>Road Anomaly Detection Report</h2>
                <p><strong>Report ID:</strong> {report_data['report_id']}</p>
                <p><strong>Timestamp:</strong> {report_data['timestamp']}</p>
                <p><strong>Severity:</strong> <span style="color: red;">{report_data['severity_level'].upper()}</span></p>
                <p><strong>Number of Detections:</strong> {report_data['num_detections']}</p>
                
                <h3>Location</h3>
                <p><strong>Coordinates:</strong> {report_data['latitude']:.6f}, {report_data['longitude']:.6f}</p>
                <p><a href="{maps_link}">View on Google Maps</a></p>
                
                <h3>Detected Anomalies</h3>
                <table border="1" cellpadding="5">
                  <tr>
                    <th>Type</th>
                    <th>Confidence</th>
                    <th>Timestamp</th>
                  </tr>
            """
            
            for det in detections:
                html_body += f"""
                  <tr>
                    <td>{det['class_name']}</td>
                    <td>{det['confidence']:.2%}</td>
                    <td>{det['timestamp']}</td>
                  </tr>
                """
            
            html_body += """
                </table>
                
                <p><em>This is an automated report from the Road Anomaly Detection System.</em></p>
              </body>
            </html>
            """
            
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            
            print(f"📧 Email sent to {recipient_email}")
            
        except Exception as e:
            print(f"❌ Email send failed: {e}")
    
    def _send_webhook_report(self, report_data, detections):
        """Send webhook report to API"""
        
        try:
            webhook_url = self.config.get('webhook_url', '')
            
            if not webhook_url:
                print("⚠️ Webhook URL not configured")
                return
            
            # Prepare payload
            payload = {
                'report_id': report_data['report_id'],
                'timestamp': report_data['timestamp'],
                'severity': report_data['severity_level'],
                'location': {
                    'latitude': report_data['latitude'],
                    'longitude': report_data['longitude']
                },
                'detections': [
                    {
                        'type': d['class_name'],
                        'confidence': d['confidence'],
                        'timestamp': d['timestamp']
                    }
                    for d in detections
                ]
            }
            
            # Send POST request
            response = requests.post(
                webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"📡 Webhook sent successfully")
            else:
                print(f"⚠️ Webhook failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Webhook send failed: {e}")
    
    def get_statistics(self):
        """Get reporting statistics"""
        return self.database.get_statistics()


def test_reporting_agent():
    """Test reporting agent"""
    
    print("="*60)
    print("🧪 Testing Reporting Agent")
    print("="*60)
    
    # Initialize agent
    agent = ReportingAgent(
        db_path='test_detections.db',
        config={
            'min_confidence': 0.5,
            'report_interval': 10,  # 10 seconds for testing
            'email_enabled': False,
            'webhook_enabled': False
        }
    )
    
    agent.start()
    
    # Simulate some detections
    print("\n📊 Simulating detections...")
    for i in range(5):
        detection = {
            'class_id': i % 4,
            'class_name': ['Pothole', 'Alligator Crack', 'Longitudinal Crack', 'Other Damage'][i % 4],
            'confidence': 0.7 + (i * 0.05),
            'bbox': [100, 100, 200, 200],
            'timestamp': datetime.now().isoformat(),
            'gps': {
                'latitude': 28.6139 + (i * 0.001),
                'longitude': 77.2090 + (i * 0.001),
                'altitude': 216.0,
                'speed_kmh': 30.0,
                'heading': 90.0,
                'fix': True
            }
        }
        
        agent.queue_detection(detection)
        print(f"  Queued: {detection['class_name']} ({detection['confidence']:.2f})")
        time.sleep(0.5)
    
    # Wait for processing
    print("\n⏳ Waiting for report generation...")
    time.sleep(15)
    
    # Get statistics
    print("\n📊 Statistics:")
    stats = agent.get_statistics()
    for stat in stats:
        print(f"  {stat['class_name']}: {stat['count']} detections")
    
    agent.stop()
    print("\n✅ Reporting Agent test complete")


if __name__ == "__main__":
    test_reporting_agent()

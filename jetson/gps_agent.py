"""
GPS Tagging Agent for Road Anomaly Detection
Captures GPS coordinates and associates them with detections
"""

import time
import json
from datetime import datetime
from pathlib import Path
import threading
import queue


# Try to import GPS library
try:
    import gpsd
    GPS_AVAILABLE = True
except ImportError:
    GPS_AVAILABLE = False
    print("⚠️ gpsd library not available. Using mock GPS.")


class GPSAgent:
    """
    GPS Agent for geo-tagging road anomalies
    """
    
    def __init__(self, port='/dev/ttyUSB0', baudrate=9600, use_mock=False):
        """
        Args:
            port: Serial port for GPS module
            baudrate: Baud rate
            use_mock: Use mock GPS data for testing
        """
        self.port = port
        self.baudrate = baudrate
        self.use_mock = use_mock or not GPS_AVAILABLE
        
        # Current GPS data
        self.current_location = {
            'latitude': 0.0,
            'longitude': 0.0,
            'altitude': 0.0,
            'speed': 0.0,
            'heading': 0.0,
            'timestamp': None,
            'fix': False
        }
        
        # Threading
        self.running = False
        self.thread = None
        self.update_interval = 1.0  # seconds
        
        # Mock GPS (for testing without hardware)
        self.mock_location = {
            'latitude': 28.6139,  # Delhi coordinates
            'longitude': 77.2090,
            'altitude': 216.0
        }
        
        # Initialize connection
        self._initialize()
    
    def _initialize(self):
        """Initialize GPS connection"""
        
        if self.use_mock:
            print("📍 GPS Agent: Using mock GPS data")
            self.current_location.update(self.mock_location)
            self.current_location['fix'] = True
            return
        
        try:
            # Connect to gpsd
            gpsd.connect()
            print(f"📍 GPS Agent: Connected to GPS on {self.port}")
            
            # Wait for fix
            print("⏳ Waiting for GPS fix...")
            timeout = 30  # seconds
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                packet = gpsd.get_current()
                if packet.mode >= 2:  # 2D or 3D fix
                    print("✅ GPS fix acquired")
                    self.current_location['fix'] = True
                    break
                time.sleep(1)
            
            if not self.current_location['fix']:
                print("⚠️ GPS fix not acquired, switching to mock mode")
                self.use_mock = True
                self.current_location.update(self.mock_location)
                self.current_location['fix'] = True
                
        except Exception as e:
            print(f"⚠️ GPS initialization failed: {e}")
            print("   Switching to mock GPS mode")
            self.use_mock = True
            self.current_location.update(self.mock_location)
            self.current_location['fix'] = True
    
    def start(self):
        """Start GPS update thread"""
        
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        
        print("📍 GPS Agent: Started")
    
    def stop(self):
        """Stop GPS update thread"""
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        
        print("📍 GPS Agent: Stopped")
    
    def _update_loop(self):
        """Continuous GPS update loop"""
        
        while self.running:
            try:
                self._update_location()
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"⚠️ GPS update error: {e}")
                time.sleep(self.update_interval)
    
    def _update_location(self):
        """Update current GPS location"""
        
        if self.use_mock:
            # Simulate slight movement for testing
            import random
            self.current_location['latitude'] += random.uniform(-0.0001, 0.0001)
            self.current_location['longitude'] += random.uniform(-0.0001, 0.0001)
            self.current_location['speed'] = random.uniform(20, 40)  # km/h
            self.current_location['heading'] = random.uniform(0, 360)
            self.current_location['timestamp'] = datetime.now().isoformat()
            return
        
        try:
            packet = gpsd.get_current()
            
            if packet.mode >= 2:  # Valid fix
                self.current_location = {
                    'latitude': packet.lat,
                    'longitude': packet.lon,
                    'altitude': packet.alt if packet.mode >= 3 else 0.0,
                    'speed': packet.hspeed,  # m/s
                    'heading': packet.track,
                    'timestamp': datetime.now().isoformat(),
                    'fix': True
                }
        except Exception as e:
            print(f"⚠️ GPS read error: {e}")
    
    def get_current_location(self):
        """
        Get current GPS coordinates
        
        Returns:
            dict: Current location data
        """
        return self.current_location.copy()
    
    def tag_detection(self, detection):
        """
        Tag a detection with GPS coordinates
        
        Args:
            detection: Detection dictionary
            
        Returns:
            Detection dictionary with GPS data
        """
        location = self.get_current_location()
        
        detection['gps'] = {
            'latitude': location['latitude'],
            'longitude': location['longitude'],
            'altitude': location['altitude'],
            'speed_kmh': location['speed'] * 3.6 if not self.use_mock else location['speed'],
            'heading': location['heading'],
            'timestamp': location['timestamp'],
            'fix': location['fix']
        }
        
        return detection
    
    def get_location_string(self):
        """Get human-readable location string"""
        
        loc = self.current_location
        return (
            f"Lat: {loc['latitude']:.6f}, "
            f"Lon: {loc['longitude']:.6f}, "
            f"Speed: {loc['speed']*3.6:.1f} km/h"
        )


class GeoFenceAgent:
    """
    Geo-fencing agent to filter detections by location
    """
    
    def __init__(self, geofence_config=None):
        """
        Args:
            geofence_config: List of geofence polygons
        """
        self.geofences = geofence_config or []
    
    def is_inside_geofence(self, latitude, longitude):
        """
        Check if location is inside any geofence
        
        Args:
            latitude: Latitude
            longitude: Longitude
            
        Returns:
            bool: True if inside any geofence
        """
        if not self.geofences:
            return True  # No geofence = all locations valid
        
        for geofence in self.geofences:
            if self._point_in_polygon(latitude, longitude, geofence):
                return True
        
        return False
    
    def _point_in_polygon(self, lat, lon, polygon):
        """
        Check if point is inside polygon (ray casting algorithm)
        
        Args:
            lat, lon: Point coordinates
            polygon: List of (lat, lon) tuples
            
        Returns:
            bool: True if point is inside polygon
        """
        n = len(polygon)
        inside = False
        
        p1_lat, p1_lon = polygon[0]
        
        for i in range(1, n + 1):
            p2_lat, p2_lon = polygon[i % n]
            
            if lon > min(p1_lon, p2_lon):
                if lon <= max(p1_lon, p2_lon):
                    if lat <= max(p1_lat, p2_lat):
                        if p1_lon != p2_lon:
                            x_intersect = (lon - p1_lon) * (p2_lat - p1_lat)
                            x_intersect = x_intersect / (p2_lon - p1_lon) + p1_lat
                            
                            if p1_lat == p2_lat or lat <= x_intersect:
                                inside = not inside
            
            p1_lat, p1_lon = p2_lat, p2_lon
        
        return inside


def generate_maps_link(latitude, longitude):
    """
    Generate Google Maps link for location
    
    Args:
        latitude: Latitude
        longitude: Longitude
        
    Returns:
        str: Google Maps URL
    """
    return f"https://www.google.com/maps?q={latitude},{longitude}"


def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two GPS coordinates (Haversine formula)
    
    Args:
        lat1, lon1: First point
        lat2, lon2: Second point
        
    Returns:
        float: Distance in meters
    """
    from math import radians, cos, sin, asin, sqrt
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Earth radius in meters
    r = 6371000
    
    return c * r


def test_gps_agent():
    """Test GPS agent functionality"""
    
    print("="*60)
    print("🧪 Testing GPS Agent")
    print("="*60)
    
    # Initialize agent (mock mode for testing)
    gps = GPSAgent(use_mock=True)
    gps.start()
    
    print("\n📍 GPS Location Updates:")
    for i in range(5):
        time.sleep(1)
        location = gps.get_current_location()
        print(f"\n[{i+1}] Location:")
        print(f"   Lat: {location['latitude']:.6f}")
        print(f"   Lon: {location['longitude']:.6f}")
        print(f"   Speed: {location['speed']:.1f} km/h")
        print(f"   Maps: {generate_maps_link(location['latitude'], location['longitude'])}")
    
    # Test detection tagging
    print("\n🏷️ Testing Detection Tagging:")
    sample_detection = {
        'class_id': 0,
        'class_name': 'Pothole',
        'confidence': 0.85,
        'bbox': [100, 150, 200, 250]
    }
    
    tagged_detection = gps.tag_detection(sample_detection)
    print(json.dumps(tagged_detection, indent=2))
    
    gps.stop()
    print("\n✅ GPS Agent test complete")


if __name__ == "__main__":
    test_gps_agent()

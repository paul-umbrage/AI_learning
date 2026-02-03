#!/usr/bin/env python3
"""
Test script to check if Redis is working properly.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from redis_cache import get_redis_cache, get_cache_stats
import redis

def test_redis_connection():
    """Test basic Redis connectivity"""
    print("=" * 60)
    print("Redis Connection Test")
    print("=" * 60)
    
    # Test 1: Check if redis package is installed
    try:
        import redis
        print("✓ Redis Python package is installed")
    except ImportError:
        print("✗ Redis Python package is NOT installed")
        print("  Install with: pip install redis")
        return False
    
    # Test 2: Check Redis cache initialization
    cache = get_redis_cache()
    if cache.enabled:
        print("✓ Redis cache is enabled")
    else:
        print("✗ Redis cache is NOT enabled")
        print("  Check your Redis connection settings")
        return False
    
    # Test 3: Test basic operations
    print("\nTesting basic Redis operations...")
    
    try:
        # Test PING
        result = cache.redis_client.ping()
        print(f"✓ PING: {result}")
        
        # Test SET/GET
        test_key = "test:connection"
        test_value = "Hello Redis!"
        cache.set(test_key, test_value, ttl=60)
        retrieved = cache.get(test_key)
        
        if retrieved == test_value:
            print(f"✓ SET/GET: Successfully stored and retrieved '{test_value}'")
        else:
            print(f"✗ SET/GET: Failed - expected '{test_value}', got '{retrieved}'")
            return False
        
        # Clean up
        cache.delete(test_key)
        print("✓ DELETE: Successfully deleted test key")
        
        # Test EXISTS
        exists_before = cache.exists(test_key)
        cache.set(test_key, "test", ttl=60)
        exists_after = cache.exists(test_key)
        cache.delete(test_key)
        
        if not exists_before and exists_after:
            print("✓ EXISTS: Key existence check works correctly")
        else:
            print("✗ EXISTS: Key existence check failed")
            return False
        
    except Exception as e:
        print(f"✗ Error during operations: {e}")
        return False
    
    # Test 4: Get cache statistics
    print("\nCache Statistics:")
    stats = get_cache_stats()
    if stats.get("enabled"):
        print(f"  Enabled: {stats.get('enabled')}")
        print(f"  Connected Clients: {stats.get('connected_clients', 'N/A')}")
        print(f"  Used Memory: {stats.get('used_memory_human', 'N/A')}")
        print(f"  Total Keys: {stats.get('total_keys', 'N/A')}")
        print(f"  Keyspace Hits: {stats.get('keyspace_hits', 'N/A')}")
        print(f"  Keyspace Misses: {stats.get('keyspace_misses', 'N/A')}")
    else:
        print(f"  {stats.get('message', 'Cache not enabled')}")
    
    # Test 5: Test direct connection with environment variables
    print("\n" + "=" * 60)
    print("Direct Connection Test")
    print("=" * 60)
    
    try:
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        redis_db = int(os.getenv("REDIS_DB", 0))
        redis_password = os.getenv("REDIS_PASSWORD", None)
        
        print(f"Connection settings:")
        print(f"  Host: {redis_host}")
        print(f"  Port: {redis_port}")
        print(f"  DB: {redis_db}")
        print(f"  Password: {'***' if redis_password else 'None'}")
        
        direct_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            socket_connect_timeout=5,
            socket_timeout=5
        )
        
        ping_result = direct_client.ping()
        print(f"✓ Direct connection PING: {ping_result}")
        
        # Get server info
        info = direct_client.info('server')
        print(f"✓ Redis Version: {info.get('redis_version', 'Unknown')}")
        print(f"✓ Redis Mode: {info.get('redis_mode', 'Unknown')}")
        
        direct_client.close()
        
    except redis.ConnectionError as e:
        print(f"✗ Direct connection failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure Redis is running: docker-compose up -d redis")
        print("  2. Check if Redis is accessible on the configured host/port")
        print("  3. Verify your REDIS_HOST and REDIS_PORT environment variables")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All Redis tests passed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_redis_connection()
    sys.exit(0 if success else 1)

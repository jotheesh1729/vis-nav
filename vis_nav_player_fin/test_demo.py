"""
Debug script to test if modules are working
Run this BEFORE demo_player.py to diagnose issues
"""

import cv2
import numpy as np
import os
from config_demo import *

print("="*70)
print("DEMO SYSTEM DEBUG TEST")
print("="*70)

# Test 1: Check directories
print("\n[1/5] Checking directories...")
if os.path.exists(DEMO_DIR):
    print(f"   OK: {DEMO_DIR} exists")
else:
    print(f"   Creating {DEMO_DIR}...")
    os.makedirs(DEMO_DIR, exist_ok=True)

if os.path.exists(DATA_DIR):
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.jpg')]
    print(f"   OK: {DATA_DIR} has {len(files)} images")
else:
    print(f"   ERROR: {DATA_DIR} not found!")
    print(f"   Create this directory and add images")

if os.path.exists(SUPERGLUE_PATH):
    print(f"   OK: {SUPERGLUE_PATH} exists")
else:
    print(f"   ERROR: {SUPERGLUE_PATH} not found!")
    print(f"   Clone SuperGlue first")

# Test 2: SuperPoint module
print("\n[2/5] Testing SuperPoint module...")
try:
    from superpoint_display import SuperPointDisplay
    sp = SuperPointDisplay()
    if sp.enabled:
        print("   OK: SuperPoint module loaded and enabled")
        
        # Test with dummy frames
        frame1 = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        frame2 = frame1.copy()
        frame2[100:150, 100:150] += 10
        
        sp.process_and_display(frame1, "Debug Test")
        sp.process_and_display(frame2, "Debug Test")
        
        print("   OK: SuperPoint processing works")
        cv2.destroyAllWindows()
    else:
        print("   ERROR: SuperPoint not enabled")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 3: ResNet module
print("\n[3/5] Testing ResNet module...")
try:
    from resnet_localization import ResNetLocalizer
    resnet = ResNetLocalizer()
    print("   OK: ResNet module loaded")
    
    # Test feature extraction
    test_img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    feat = resnet.extract_feature(test_img)
    print(f"   OK: Feature extraction works (dim: {len(feat)})")
    
    # Test database loading
    if os.path.exists(DATA_DIR):
        print("   Loading database (this may take a while)...")
        resnet.load_database()
        if resnet.database_features is not None:
            print(f"   OK: Database loaded ({len(resnet.database_features)} features)")
        else:
            print("   WARNING: Database not loaded")
    else:
        print("   SKIP: No database directory")
        
except Exception as e:
    print(f"   ERROR: {e}")

# Test 4: Frame saving
print("\n[4/5] Testing frame saving...")
try:
    test_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    test_path = os.path.join(DEMO_DIR, "test_frame.jpg")
    
    success = cv2.imwrite(test_path, test_frame)
    
    if success and os.path.exists(test_path):
        print(f"   OK: Frame saved to {test_path}")
        # Clean up
        os.remove(test_path)
        print("   OK: Frame saving works")
    else:
        print("   ERROR: Could not save frame")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 5: Config
print("\n[5/5] Checking configuration...")
print(f"   RECORD_EVERY_FRAME: {RECORD_EVERY_FRAME}")
print(f"   SHOW_FEATURE_MATCHES: {SHOW_FEATURE_MATCHES}")
print(f"   SHOW_NEXT_BEST_VIEW: {SHOW_NEXT_BEST_VIEW}")
print(f"   PLAYBACK_FPS: {PLAYBACK_FPS}")
print(f"   USE_GPU: {USE_GPU}")

# Summary
print("\n" + "="*70)
print("DEBUG TEST COMPLETE")
print("="*70)

issues = []
if not os.path.exists(DATA_DIR):
    issues.append("- Data directory not found")
if not os.path.exists(SUPERGLUE_PATH):
    issues.append("- SuperGlue not found")

if len(issues) > 0:
    print("\nISSUES FOUND:")
    for issue in issues:
        print(issue)
    print("\nFix these before running demo_player.py")
else:
    print("\nAll checks passed! Ready to run demo_player.py")

print("="*70)
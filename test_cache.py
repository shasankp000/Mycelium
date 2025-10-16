"""
Test script to verify calibration caching works correctly.
This should load cached metrics without recomputing.
"""

from unified_expert_system import UnifiedExpertSystem

print("="*60)
print("🧪 TESTING CALIBRATION CACHE")
print("="*60)
print("\nInitializing expert system...")
print("(If cache works, BERT models should NOT load)\n")

expert_system = UnifiedExpertSystem()

print("\n" + "="*60)
print("📊 CACHE STATUS")
print("="*60)

for domain, expert in expert_system.experts.items():
    if hasattr(expert, 'get_calibration_cache_info'):
        info = expert.get_calibration_cache_info()
        print(f"\n🔧 {domain.upper()} Expert:")
        print(f"   Cache exists: {info['exists']}")
        if info['exists']:
            print(f"   Validation accuracy: {info['validation_accuracy']:.4f}")
            print(f"   Samples used: {info['num_samples']}")
            print(f"   Method: {info['method']}")
            print(f"   Timestamp: {info['timestamp']}")
            print(f"   Fingerprint: {info['fingerprint_hash']}")
    else:
        print(f"\n🔧 {domain.upper()} Expert:")
        print(f"   Type: SVM (no cache needed)")
        print(f"   Calibration score: {expert.calibration_score:.4f}")

print("\n" + "="*60)
print("✅ TEST COMPLETE")
print("="*60)

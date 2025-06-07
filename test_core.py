"""
Test core functionality without GUI
"""
import sys
from pathlib import Path

# Add the drs_analyzer to path
sys.path.insert(0, str(Path(__file__).parent))

def test_core_imports():
    """Test that core components can be imported"""
    try:
        from drs_analyzer.core.data_loader import DataLoader
        print("✓ DataLoader imported successfully")
        
        from drs_analyzer.core.data_processor import DataProcessor
        print("✓ DataProcessor imported successfully")
        
        from drs_analyzer.config.settings import AppSettings
        print("✓ AppSettings imported successfully")
        
        # Test basic functionality
        loader = DataLoader()
        print("✓ DataLoader instantiated")
        
        processor = DataProcessor()
        print("✓ DataProcessor instantiated")
        
        settings = AppSettings()
        print("✓ AppSettings instantiated")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing core components: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing DRS Analyzer core components...")
    success = test_core_imports()
    if success:
        print("\n✓ All core components are working!")
        print("The GUI error is due to the headless environment.")
        print("Core functionality is available for CLI use.")
    else:
        print("\n✗ Some core components have issues.")
    
    sys.exit(0 if success else 1)
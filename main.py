"""
main.py - Entry point for Naval Combat Simulation
Complete implementation with all requested features
"""

import sys
from PyQt6.QtWidgets import QApplication
from ui import NavalSimApp

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    print("="*60)
    print("🚀 NAVAL COMBAT SIMULATION - COMPLETE v2.1")
    print("="*60)
    print("📋 Features Implemented:")
    print("   ✓ Military-style UI (app2.py)")
    print("   ✓ AI-adaptive ML scenarios (app3_ml.py)")
    print("   ✓ Comms Link window")
    print("   ✓ Status Log window")
    print("   ✓ Range-based interception (100m)")
    print("   ✓ Auto-scanning (150m range)")
    print("   ✓ Hidden threat levels")
    print("   ✓ Scrollable vessel details")
    print("   ✓ Fixed start menu")
    print("   ✓ Mark Threat/Safe buttons")
    print("="*60)
    print()
    
    window = NavalSimApp()
    window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()

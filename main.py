"""
main.py - Entry point for Naval Combat Simulation
Clean implementation with proper separation of UI and backend
"""

import sys
from PyQt6.QtWidgets import QApplication
from ui import NavalSimApp


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    print("="*60)
    print("🚀 NAVAL COMBAT SIMULATION - REFACTORED v3.0")
    print("="*60)
    print("📋 Architecture:")
    print("   ✓ Clean separation: UI / Backend")
    print("   ✓ Backend: All game logic and state")
    print("   ✓ UI: Pure visualization and input")
    print("   ✓ Communication via controller methods")
    print()
    print("📋 Features:")
    print("   ✓ Military-style radar UI")
    print("   ✓ WASD movement controls")
    print("   ✓ Patrol phase with zone expansion")
    print("   ✓ Vessel hailing and communication")
    print("   ✓ Threat detection and classification")
    print("   ✓ Intercept, mark safe/threat actions")
    print("   ✓ Distress call system")
    print("   ✓ Status log and reports")
    print("   ✓ Dynamic enemy AI movement")
    print("="*60)
    print()
    
    window = NavalSimApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
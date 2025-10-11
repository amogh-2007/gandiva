"""
main.py - Entry point for Enhanced Naval Combat Simulation
Updated to work with dynamic zone management and enhanced communication system
"""

import sys
from PyQt6.QtWidgets import QApplication
from ui import NavalSimApp

def main():
    """Enhanced main application entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    print("="*70)
    print("🚀 ENHANCED NAVAL COMBAT SIMULATION - v4.0")
    print("="*70)
    print("📋 NEW FEATURES:")
    print(" ✓ Dynamic zone shrinking when boat moves out of patrol area")
    print(" ✓ Communication center moved to bottom panel with popup window")
    print(" ✓ Enhanced distress call system with comprehensive reporting")
    print(" ✓ AI-powered threat analysis and vessel communication")
    print(" ✓ Real-time zone status visualization")
    print(" ✓ Improved tactical interface and controls")
    print()
    print("📋 CORE FEATURES:")
    print(" ✓ Military-style radar UI with dynamic zone management")
    print(" ✓ WASD movement controls with space to stop")
    print(" ✓ Patrol phase with intelligent zone expansion/contraction")
    print(" ✓ Enhanced vessel hailing and communication system")
    print(" ✓ Advanced threat detection and classification")
    print(" ✓ Intercept, mark safe/threat actions")
    print(" ✓ Comprehensive distress call and backup system")
    print(" ✓ Real-time status logging and tactical reports")
    print(" ✓ Dynamic AI-controlled enemy vessel movement")
    print("="*70)
    print()

    window = NavalSimApp()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
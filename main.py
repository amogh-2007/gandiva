"""
main.py - Entry point for Enhanced Naval Combat Simulation
Fixed radar system with proper zone expansion/shrinking and boundary enforcement
"""

import sys
from PyQt6.QtWidgets import QApplication
from ui import NavalSimApp

def main():
    """Enhanced main application entry point with fixed radar system"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    print("="*70)
    print("🚀 NAVAL COMBAT SIMULATION - FIXED RADAR v2.5")
    print("="*70)
    print("📋 RADAR FIXES:")
    print(" ✓ Sky-blue radar background")
    print(" ✓ Green player boat (triangle)")
    print(" ✓ Zone expands to FULL SCREEN when entered")
    print(" ✓ Zone shrinks back to original when exited")
    print(" ✓ AI boats spawn across entire screen in zone")
    print(" ✓ Player boat NEVER goes out of bounds")
    print(" ✓ AI boats NEVER go out of bounds")
    print(" ✓ Red zone only visible when OUTSIDE")
    print()
    print("📋 CORE FEATURES:")
    print(" ✓ Military-style radar UI with dynamic zone management")
    print(" ✓ WASD movement controls with SPACE to stop")
    print(" ✓ Patrol phase with intelligent zone expansion/contraction")
    print(" ✓ Enhanced vessel hailing and communication system")
    print(" ✓ Advanced threat detection and classification")
    print(" ✓ Intercept, mark safe/threat actions")
    print(" ✓ Comprehensive distress call and backup system")
    print(" ✓ Real-time status logging and tactical reports")
    print(" ✓ Dynamic AI-controlled enemy vessel movement")
    print()
    print("🎮 CONTROLS:")
    print(" • W/A/S/D - Move player boat")
    print(" • SPACE - Stop all movement")
    print(" • Click vessels - Hail and interact")
    print(" • Enter RED ZONE - Mission begins, zone expands")
    print(" • Exit zone - Zone shrinks back, vessels disappear")
    print("="*70)
    print()
    print("🚀 Launching simulation...")
    print()

    window = NavalSimApp()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
import sys
from PySide6.QtWidgets import QApplication
from ui.login_window import LoginWindow

def main():
    app = QApplication(sys.argv)
    # Apply dark futuristic style
    app.setStyleSheet(open('assets/style.qss', 'r').read())
    login = LoginWindow()
    login.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

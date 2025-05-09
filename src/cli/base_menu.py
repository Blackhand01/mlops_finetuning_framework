from abc import ABC, abstractmethod

class BaseMenu(ABC):
    """Abstract base class for all menus."""

    @abstractmethod
    def show(self):
        """Display the menu and handle user interaction."""
        pass

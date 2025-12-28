"""This module implements a simple observer pattern for monitoring progress updates."""

class Observer:
    """A simple observer class to monitor progress updates."""
    def __init__(self) -> None:
        """Initializes the observer with an empty list of observers."""
        self.liste = list()

    def add_observer(self, ob: 'Observer') -> None:
        """Adds an observer to the list."""
        self.liste.append(ob)

    def del_observer(self, ob: 'Observer'):
        """Removes an observer from the list."""
        if ob in self.liste:
            self.liste.remove(ob)

    def notify_observer(self, value: int) -> None:
        """Notifies all observers with the given value."""
        for ob in self.liste:
            ob.update(value)
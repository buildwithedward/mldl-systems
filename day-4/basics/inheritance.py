from abc import ABC, abstractmethod

class Payment(ABC):
    @abstractmethod
    def process_payment(self, amount: float) -> None:
        pass

class CreditCard(Payment):
    def process_payment(self, amount: float) -> None:
        print(f"Processing CC payment of ${amount}")


class UPI(Payment):
    def process_payment(self, amount: float) -> None:
        print(f"Processing UPI payment of ${amount}")

c = CreditCard()
c.process_payment(100)

u = UPI()
u.process_payment(200)
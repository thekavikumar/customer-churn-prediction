from pydantic import BaseModel

class Customer(BaseModel):
    customerID: str
    gender: str
    
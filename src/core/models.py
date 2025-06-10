from pydantic import BaseModel

class MemInfo(BaseModel):
    total : float
    free : float
    percent : float

class RAMInfo(MemInfo):
    available : float

class VRAMInfo(MemInfo):
    allocated : float
    cached : float
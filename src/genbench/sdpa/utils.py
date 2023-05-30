from pydantic import BaseModel


class SDPARecord(BaseModel):
    batch_size: int
    seqlen: int
    headdim: int
    nheads: int
    time_pt_eager: float
    time_pt_native: float
    time_flash: float
    time_mem_eff: float
    time_math: float

plnt_cds = {
    "A_DS": "hn_ds",
    "A_WB": "hn_wb",
    "A_GY": "hn_gy",
    "A_IS": "hn_is",
    "A_PJ": "hn_pj",
    "B_HJ": "hn_hj",
    "B_SJ": "hn_so",
    "A_BW": "hs_bw",
    "A_SH": "hs_sh",
    "A_SN": "hs_sn",
    "A_SJ": "hs_sj",
    "C_CU": "hs_cu",
    "D_DH": "ys_dh",
    "E_HS": "ys_hs",
    "E_DB": "ys_db",
    "E_PL": "ys_pl",
    "E_DJ": "ys_dj",
    "E_BY": "ys_by",
}


class AgentConfig:
    def __init__(self, location_code="A", plant_code="SN", algorithm_code="C"):
        self.location_code = location_code
        self.plant_code = plant_code
        self.location_plant_code = f"{self.location_code}_{self.plant_code}"
        self.algorithm_code = algorithm_code
        self.plnt_cd = plnt_cds[f"{self.location_code}_{self.plant_code}"]

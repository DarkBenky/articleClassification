"""Helpers shared by the training scripts in code/."""

import csv

from project_config import CAMEO_CODES_PATH, FIPS_CODES_PATH


def getFipCodes():
    with open(FIPS_CODES_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fipCodes = {}
        for row in reader:
            fips = row["FIPS 10-4"].strip()
            countryName = row["Name"].strip()
            if fips and fips != "-":
                fipCodes[fips] = countryName
    return list(fipCodes.keys()), list(fipCodes.values())


def getCameoCodes():
    with open(CAMEO_CODES_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        cameoCodes = {}
        for row in reader:
            if len(row) != 2:
                continue
            code = row[0].strip()
            cameoName = row[1].strip()
            if code and cameoName:
                cameoCodes[code] = cameoName
    return list(cameoCodes.keys()), list(cameoCodes.values())


def countLines(path):
    total = 0
    lastByte = b""
    with open(path, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            total += chunk.count(b"\n")
            lastByte = chunk[-1:]
    return total + (1 if lastByte and lastByte != b"\n" else 0)

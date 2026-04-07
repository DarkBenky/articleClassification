def codeToName(code):
    with open('fipsCodes.csv', 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if parts[0] == code:
                return parts[2]
    return code
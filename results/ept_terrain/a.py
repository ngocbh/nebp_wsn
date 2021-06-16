import os

name_map = {
    'NIn24_medium_uu-dem7_r25_1_40':'NIn25_medium_uu-dem7_r25_1_40',
    'NIn25_medium_uu-dem8_r25_1_40':'NIn26_medium_uu-dem8_r25_1_40',
    'NIn25_medium_uu-dem9_r25_1_40':'NIn27_medium_uu-dem9_r25_1_40',
    'NIn26_medium_uu-dem10_r25_1_40':'NIn28_medium_uu-dem10_r25_1_40' 
}

for d1 in os.listdir('./'):
    if '1.8' not in d1:
        continue
    print(d1)
    for key, value in name_map.items():
        print(key, '->', value)
        os.rename(os.path.join(d1, key), os.path.join(d1, value))
        # if 'NIn20' in d2:
            # print(d2)
            # with open(os.path.join(f"{d1}/{d2}", "P.txt"), mode='w') as f:
                # f.write("1 0.9572\n100 0.0008")

	

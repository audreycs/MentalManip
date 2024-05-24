import csv

nonmanip_data_list = []
with open('../dataset/new_processed_mentalmanip_con_final.csv', 'r', newline='', encoding='utf-8') as infile:
    content = csv.reader(infile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    columns = None
    for idx, row in enumerate(content):
        if idx == 0:
            columns = row
            continue
        if row[2] == '0' or row[2] == 0:
            nonmanip_data_list.append(row)
        else:
            continue

with open('../dataset/new_processed_mentalmanip_con_final_nonmanip.csv', 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(columns)
    for idx in nonmanip_data_list:
        writer.writerow(idx)

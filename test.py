from prettytable import PrettyTable

table = PrettyTable()

# table header
table.field_names = ["이름", "나이"]

# add table row
table.add_row(["nahye", 26])
table.add_row(["henna", 30])

print(table)
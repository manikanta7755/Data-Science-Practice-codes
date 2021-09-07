#Applying association rules for Books data set
#Installing packages

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

#Importing file into python
books = pd.read_csv("G:\\Mani\\association rules\\book.csv")

#Applying association rules using apriori algorithm
frequent_itemsets = apriori(books, min_support = 0.002, max_len = 2, use_colnames = True)

frequent_itemsets.sort_values('support', ascending = False, inplace = True)

#Plotting the most frequent data
plt.bar(x = list(range(0, 11)), height = frequent_itemsets.support[0:11], color ='rgmyk')
plt.xticks(list(range(0, 11)), frequent_itemsets.itemsets[0:11], rotation=20)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(20)

rules.sort_values('lift', ascending = False).head(10)

#listing of most frequent occured data
def to_list(i):
    return (sorted(list(i)))

ma_books = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_books = ma_books.apply(sorted)

rules_sets = list(ma_books)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting association rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift', ascending = False).head(10)
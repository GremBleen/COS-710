import pandas as pd
from assign1.gp_class import GPClass

data = pd.read_csv("192_vineyard.tsv", sep="\t")

gp = GPClass()

gp.fit(data, config_file="default_config.json")

gp.train()

gp.test()
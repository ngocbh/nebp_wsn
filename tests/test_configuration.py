import yaml
from yaml import Loader
import json
parameters = yaml.load(open('./configs/_configurations.yml', mode='r'), Loader=Loader)

print(parameters)
print(json.dumps(parameters, indent=4))

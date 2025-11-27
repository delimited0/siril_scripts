# Quirks

1. In Siril calibrate command, the argument for synthetic bias must be int, not float. So cast floats to int if you are calling from sirilpy.
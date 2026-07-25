"""
Shared application state — holds the loaded prediction service instance.
Populated at startup in main.py, read by routes.py.
"""

service = None
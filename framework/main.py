import logging
from core.app import MainFramework

"""
main: caller
app: core orchestration logic
"""

if __name__ == "__main__":
    generator = MainFramework(9, 3, 25)
    generator.run()
    logging.basicConfig(level=logging.DEBUG)
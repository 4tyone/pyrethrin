import os

# Disable static checking during tests since tests intentionally test
# edge cases without proper exhaustive handling
os.environ["PYRETHRIN_DISABLE_STATIC_CHECK"] = "1"

from desc import Descriptor
import matplotlib.pyplot as plt
# Process bundle adjustment after the fact, because I have collected data already (takes ~30 mins).

descriptor = Descriptor(None)
descriptor.load_pickle("full_data_00.pkl")
a, b = descriptor.bundle_adjustment()
plt.plot(a)
plt.show()
plt.plot(b.fun)
plt.show()
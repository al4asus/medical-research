import matplotlib.pyplot as plt
import numpy as np

# Örnek veri oluşturma
x = np.linspace(0, 5, 100)
y = x**2

# Eğriyi çiz
plt.plot(x, y)

# Eğri altındaki alanı göster
plt.fill_between(x, y, color='skyblue', alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Eğri Altındaki Alan')
plt.grid(True)
plt.show()
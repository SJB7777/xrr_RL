import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class XRRSample:
    density: float
    roughness: float
    thickness: float
    absorption: float  # 추가된 흡수 계수


def fresnel_coefficients(n1: float, n2: float, theta: float) -> tuple[float, float]:
    """Calculate the Fresnel reflection and transmission coefficients."""
    cos_theta1 = np.cos(theta)
    cos_theta2 = np.sqrt(1 - (n1 / n2 * np.sin(theta))**2)
    r = (n1 * cos_theta1 - n2 * cos_theta2) / (n1 * cos_theta1 + n2 * cos_theta2)
    t = 2 * n1 * cos_theta1 / (n1 * cos_theta1 + n2 * cos_theta2)
    return r, t


def parratt_reflectivity(layers: list[XRRSample], wavelength: float, angles: np.ndarray) -> np.ndarray:
    """Calculate the reflectivity using the Parratt recursion method."""
    k0 = 2 * np.pi / wavelength
    reflectivity = np.zeros_like(angles, dtype=np.complex128)

    for i, theta in enumerate(angles):
        kz = k0 * np.sin(theta)
        r_total = 0
        for j in range(len(layers) - 1, 0, -1):
            n1 = layers[j-1].density
            n2 = layers[j].density
            sigma = layers[j].roughness
            absorption = layers[j].absorption
            r, _ = fresnel_coefficients(n1, n2, theta)
            r *= np.exp(-2 * (kz * sigma)**2)  # Apply roughness correction
            r *= np.exp(-absorption * kz)  # Apply absorption correction
            r_total = (r + r_total * np.exp(2j * kz * layers[j].thickness)) / (1 + r * r_total * np.exp(2j * kz * layers[j].thickness))
        reflectivity[i] = np.abs(r_total)**2
    
    return reflectivity


if __name__ == '__main__':
    # Example usage
    layers = [
        XRRSample(density=2.33, roughness=0.36, thickness=0.0, absorption=0.1),  # Substrate
        XRRSample(density=3.36, roughness=0.5, thickness=2.07, absorption=0.1),
        XRRSample(density=6.50, roughness=1.40, thickness=35.07, absorption=0.1),
        XRRSample(density=4.01, roughness=1.01, thickness=1.62, absorption=0.1),
    ]

    wavelength = 1.54  # Angstroms
    angles_degrees = np.linspace(0, 8, 1000)  # Degrees
    angles_radians = np.radians(angles_degrees)  # Convert to radians

    reflectivity = parratt_reflectivity(layers, wavelength, angles_radians)

    plt.plot(angles_degrees, reflectivity.real)  # Use the real part of reflectivity
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Reflectivity')
    plt.yscale('log')
    plt.show()
# =================================  TESTY  ===================================
# Testy do tego pliku obejmują jedynie weryfikację poprawności wyników dla
# prawidłowych danych wejściowych - obsługa niepoprawych danych wejściowych
# nie jest ani wymagana ani sprawdzana. W razie potrzeby lub chęci można ją 
# wykonać w dowolny sposób we własnym zakresie.
# =============================================================================
import numpy as np


def chebyshev_nodes(n: int = 10) -> np.ndarray | None:
    """Funkcja generująca wektor węzłów Czebyszewa drugiego rodzaju (n,) 
    i sortująca wynik od najmniejszego do największego węzła.

    Args:
        n (int): Liczba węzłów Czebyszewa.
    
    Returns:
        (np.ndarray): Wektor węzłów Czebyszewa (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(n, int) or n <= 1:
        return None

    k = np.arange(0, n)
    x = np.cos(k * np.pi / (n - 1))

    return x
def bar_cheb_weights(n: int = 10) -> np.ndarray | None:
    """Funkcja tworząca wektor wag dla węzłów Czebyszewa wymiaru (n,).

    Args:
        n (int): Liczba wag węzłów Czebyszewa.
    
    Returns:
        (np.ndarray): Wektor wag dla węzłów Czebyszewa (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(n, int) or n <= 1:
        return None

    j = np.arange(0, n)
    delta = np.ones(n)
    delta[0] = 0.5
    delta[-1] = 0.5

    w = (-1) ** j * delta
    return w
def barycentric_inte(
    xi: np.ndarray, yi: np.ndarray, wi: np.ndarray, x: np.ndarray
) -> np.ndarray | None:
    """Funkcja przeprowadza interpolację metodą barycentryczną dla zadanych 
    węzłów xi i wartości funkcji interpolowanej yi używając wag wi. Zwraca 
    wyliczone wartości funkcji interpolującej dla argumentów x w postaci 
    wektora (n,).

    Args:
        xi (np.ndarray): Wektor węzłów interpolacji (m,).
        yi (np.ndarray): Wektor wartości funkcji interpolowanej w węzłach (m,).
        wi (np.ndarray): Wektor wag interpolacji (m,).
        x (np.ndarray): Wektor argumentów dla funkcji interpolującej (n,).
    
    Returns:
        (np.ndarray): Wektor wartości funkcji interpolującej (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    # Walidacja wejścia
    if not isinstance(xi, np.ndarray) or not isinstance(yi, np.ndarray) \
       or not isinstance(wi, np.ndarray) or not isinstance(x, np.ndarray):
        return None
    if xi.shape != yi.shape or xi.shape != wi.shape:
        return None

    xi = xi.astype(float)
    yi = yi.astype(float)
    wi = wi.astype(float)
    x = x.astype(float)

    P = np.zeros_like(x, dtype=float)

    for i, xv in enumerate(x):
        diff = xv - xi
        mask = np.isclose(diff, 0.0)

        if np.any(mask):  
            P[i] = yi[mask][0]
        else:
            num = np.sum(wi * yi / diff)
            den = np.sum(wi / diff)
            P[i] = num / den

    return P

def L_inf(
    xr: int | float | list | np.ndarray, x: int | float | list | np.ndarray
) -> float | None:
    """Funkcja obliczająca normę L-nieskończoność. Powinna działać zarówno na 
    wartościach skalarnych, listach, jak i wektorach biblioteki numpy.

    Args:
        xr (int | float | list | np.ndarray): Wartość dokładna w postaci 
            skalara, listy lub wektora (n,).
        x (int | float | list | np.ndarray): Wartość przybliżona w postaci 
            skalara, listy lub wektora (n,).

    Returns:
        (float): Wartość normy L-nieskończoność.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(xr, (int, float, list,np.ndarray )) or not isinstance(x,(int, float, list,np.ndarray ) ):
        return None
    xr = np.array(xr)
    x = np.array(x)

    if xr.shape != x.shape:
        return None

    linf = np.max(np.abs(xr - x))
    return linf
#funkcje do zadania 2
def f1(x):
    return np.sign(x)*x+x**2
def f2(x):
    return np.sign(x)*x**2
def f3(x):
    return abs(np.sin(5*x))**3
def f4a(x):
    return 1/(1+x**2)
def f4b(x):
    return 1/(1+25*x**2)
def f4c(x):
    return 1/(1+100*x**2)
def f5(x):
    return np.sign(x)
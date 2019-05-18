import lhapdf
import math
import numpy as np 
from random import sample

# Get the PDF4LHC15 for test purpose and print some description
pdf = lhapdf.getPDFSet("NNPDF31_nnlo_as_0118")
print(pdf.description)
pdf_central   = pdf.mkPDF(0)

# Define the scale 
Q_pdf = 1.7874388

# Nodes from LHAPDF
x_nodes = "1.0000000e-09 1.2805087e-09 1.6397027e-09 2.0996536e-09 2.6886248e-09 3.4428076e-09 4.4085452e-09 5.6451808e-09 7.2287034e-09 9.2564179e-09 1.1852924e-08 1.5177773e-08 1.9435271e-08 2.4887035e-08 3.1868066e-08 4.0807337e-08 5.2254152e-08 6.6911899e-08 8.5681272e-08 1.0971562e-07 1.4049181e-07 1.7990099e-07 2.3036479e-07 2.9498413e-07 3.7772976e-07 4.8368627e-07 6.1936450e-07 7.9310166e-07 1.0155736e-06 1.3004509e-06 1.6652388e-06 2.1323528e-06 2.7304964e-06 3.4964246e-06 4.4772022e-06 5.7330966e-06 7.3412804e-06 9.4005738e-06 1.2037517e-05 1.5414146e-05 1.9737949e-05 2.5274616e-05 3.2364367e-05 4.1442855e-05 5.3067938e-05 6.7953959e-05 8.7015639e-05 1.1142429e-04 1.4267978e-04 1.8270270e-04 2.3395241e-04 2.9957810e-04 3.8361238e-04 4.9121901e-04 6.2901024e-04 8.0545312e-04 1.0313898e-03 1.3207036e-03 1.6911725e-03 2.1655612e-03 2.7730201e-03 3.5508765e-03 4.5469285e-03 5.8223817e-03 7.4556107e-03 9.5469747e-03 1.2224985e-02 1.5654200e-02 2.0045340e-02 2.5668233e-02 3.2868397e-02 4.2088270e-02 5.3894398e-02 6.9012248e-02 8.8370787e-02 1.0000000e-01 1.1216216e-01 1.2432432e-01 1.3648649e-01 1.4864865e-01 1.6081081e-01 1.7297297e-01 1.8513514e-01 1.9729730e-01 2.0945946e-01 2.2162162e-01 2.3378378e-01 2.4594595e-01 2.5810811e-01 2.7027027e-01 2.8243243e-01 2.9459459e-01 3.0675676e-01 3.1891892e-01 3.3108108e-01 3.4324324e-01 3.5540541e-01 3.6756757e-01 3.7972973e-01 3.9189189e-01 4.0405405e-01 4.1621622e-01 4.2837838e-01 4.4054054e-01 4.5270270e-01 4.6486486e-01 4.7702703e-01 4.8918919e-01 5.0135135e-01 5.1351351e-01 5.2567568e-01 5.3783784e-01 5.5000000e-01 5.6216216e-01 5.7432432e-01 5.8648649e-01 5.9864865e-01 6.1081081e-01 6.2297297e-01 6.3513514e-01 6.4729730e-01 6.5945946e-01 6.7162162e-01 6.8378378e-01 6.9594595e-01 7.0810811e-01 7.2027027e-01 7.3243243e-01 7.4459459e-01 7.5675676e-01 7.6891892e-01 7.8108108e-01 7.9324324e-01 8.0540541e-01 8.1756757e-01 8.2972973e-01 8.4189189e-01 8.5405405e-01 8.6621622e-01 8.7837838e-01 8.9054054e-01 9.0270270e-01 9.1486486e-01 9.2702703e-01 9.3918919e-01 9.5135135e-01 9.6351351e-01 9.7567568e-01 9.8783784e-01 1.0000000e+00"

# Construc the array of x values
sx_nodes = x_nodes.split()
x_pdf = np.array([float(x) for x in sx_nodes])


# Define the list of flavors
flavors_list = [1,2]

# Define a function which does the sampling
def sample_pdf(n=len(x_pdf)):
    data  = []
    for x in x_pdf:
        row = []
        for fl in flavors_list:
            row.append(pdf_central.xfxQ2(fl,x,Q_pdf)-pdf_central.xfxQ2(-fl,x,Q_pdf))
        data.append(row)
    return np.array(data)
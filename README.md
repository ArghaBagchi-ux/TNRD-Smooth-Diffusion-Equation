### Theory of TNRD Smooth Diffusion Equation for Image Denoising

#### Introduction
Image denoising is essential in image processing, aiming to remove noise while preserving crucial details like edges. Traditional diffusion equations have been effective but often introduce artifacts, particularly at the image boundaries. This paper explores a modified Trainable Nonlinear Reaction Diffusion (TNRD) model to mitigate these imperfections using boundary handling techniques and integrating the strengths of existing denoising models.

#### Diffusion Equations and Boundary Handling
The TNRD model applies a series of nonlinear diffusion steps to the image. In the original TNRD model, diffusion is performed using a set of trained convolution kernels and nonlinear functions. However, applying these kernels uniformly across the image can lead to artifacts at the boundaries due to the symmetric boundary conditions. 

To address this, the modified TNRD model uses padded input images to ensure that the convolution operations at the boundaries do not introduce artifacts. By padding the image with mirror reflections of itself before applying the diffusion step and cropping the central region after the diffusion, the model maintains the integrity of the image boundaries. This padding and cropping process ensures that the convolution operations behave consistently across the entire image, including the edges.

#### Mathematical Formulation and Convolution
In the context of diffusion equations, convolutions can be interpreted in multiple ways. For image denoising, the convolution of an image with a kernel can be represented as a matrix-vector product. This interpretation simplifies the calculation of gradients during the training process. The convolution operation can be efficiently implemented using matrix multiplications, which is a standard technique in convolutional neural networks (CNNs).

#### Robust Perona-Malik (RPM) Model
The RPM model introduces robustness to the denoising process by incorporating a Gaussian convolution in the diffusion coefficient. This convolution helps the model to ignore small noise-induced details and focus on significant image features. The RPM model effectively handles additive noise by slowing down diffusion at edges and speeding it up in flat regions, thus preserving important image structures.

#### Doubly Degenerated (DD) Diffusion Model
The DD diffusion model was developed to address the challenge of removing multiplicative noise, which is more complex than additive noise. This model uses a diffusion coefficient that considers both the gradient and the gray level of the image. By adapting the diffusion speed based on the local image characteristics, the DD model can effectively reduce noise while preserving important details. However, the model's edge detection function can degenerate, making theoretical analysis challenging.

#### Proposed Hybrid Model
The proposed model combines the strengths of the RPM and DD models. It utilizes local image information, including gradients and gray values, to guide the diffusion process. The model's diffusion coefficient integrates a Gaussian convolution to enhance robustness and ensure well-posedness. This hybrid approach aims to achieve efficient noise removal, especially for high-level multiplicative noise, while protecting key image features such as edges.

By carefully handling image boundaries and incorporating effective diffusion coefficients, the proposed model addresses the limitations of existing methods. It leverages convolution operations for practical and efficient training, making it suitable for various image denoising applications. This theoretical foundation provides a robust framework for developing advanced denoising techniques that maintain image quality and detail.

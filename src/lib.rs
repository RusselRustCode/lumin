use image::{open, GrayImage, ImageBuffer, Luma};
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use pyo3::prelude::*;
use numpy::{PyArray2, PyArrayMethods};
#[pyclass]
pub struct PgmProcessor {
    image: GrayImage
}

#[pymethods]
impl PgmProcessor {
    #[new]
    pub fn new_from_path(path: &str) -> PyResult<Self> {
        let img = open(path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Ошибка, не получилось открыть файл: {}", e)))?;
        let gray_img = img.to_luma8();
        Ok(Self { image: gray_img })
    }

    #[staticmethod]
    pub fn from_array(width: u32, height: u32, data: Vec<u8>) -> PyResult<Self> {
        if data.len() != (width * height) as usize {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Длина вектора не сходится "
            ));
        }
        
        let image = ImageBuffer::from_raw(width, height, data)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Неправильная размерность"))?;
        
        Ok(Self { image })
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        self.image.save(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Не получилось сохранить изображение {}", e)))
    }

    // pub fn save_as_pgm(&self, path: &str) -> PyResult<()> {
    //     save_buffer(
    //         path,
    //         &self.image,
    //         self.image.width(),
    //         self.image.height(),
    //         image::ColorType::L8
    //     ).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Не получилось сохранить в PGM: {}", e)))
    // }

    pub fn get_dimensions(&self) -> (u32, u32) {
        self.image.dimensions()
    }

    pub fn get_pixel_data(&self) -> Vec<u8> {
        self.image.as_raw().clone()
    }


    pub fn mean_filter(&self, kernel_size: usize) -> PyResult<Self> {
        let output = self.mean_filter_impl(kernel_size);
        Ok(Self { image: output })
    }


    pub fn median_filter(&self, kernel_size: usize) -> PyResult<Self> {
        let output = self.median_filter_impl(kernel_size);
        Ok(Self { image: output })
    }


    pub fn gaussian_filter(&self, sigma: f32, kernel_size: usize) -> PyResult<Self> {
        let output = self.gaussian_filter_impl(sigma, kernel_size);
        Ok(Self { image: output })
    }


    pub fn sigma_filter(&self, sigma_threshold: u8, kernel_size: usize) -> PyResult<Self> {
        let output = self.sigma_filter_impl(sigma_threshold, kernel_size);
        Ok(Self { image: output })
    }


    pub fn absolute_difference_map(&self, filtered_image: &Self) -> PyResult<Self> {
        let output = self.absolute_difference_map_impl(&filtered_image.image);
        Ok(Self { image: output })
    }


    pub fn add_gaussian_noise(&self, sigma: f32) -> PyResult<Self> {
        let output = self.add_gaussian_noise_impl(sigma);
        Ok(Self { image: output })
    }

    pub fn add_salt_pepper_noise(&self, salt_prob: f32, pepper_prob: f32) -> PyResult<Self> {
        let output = self.add_salt_pepper_noise_impl(salt_prob, pepper_prob);
        Ok(Self { image: output })
    }

    
    #[staticmethod]
    pub fn create_gauss_kernel(sigma: f32, kernel_size: usize) -> PyResult<Vec<f32>> {
        let kernel = Self::create_gauss_kernel_impl1(sigma, kernel_size);
        Ok(kernel.into_raw_vec_and_offset().0)
    }


    pub fn unsharp_mask(&self, sigma: f32, kernel_size: usize, lambda: f32) -> PyResult<Self> {
        let blurred = self.gaussian_filter_impl(sigma, kernel_size);
        let sharpened = self.unsharp_mask_impl(&blurred, lambda);
        Ok(Self { image: sharpened })
    }

    //Часть 3

    pub fn high_pass_filter(&self, coeff: f32, sigma: f32, filter: Option<&str>) -> PyResult<Self>{
        let high_pass_filter_image = self.high_pass_filter_impl(coeff, sigma, filter);
        Ok(Self {image: high_pass_filter_image})
    }


    pub fn convolution(
        &self,
        kernel_size: usize,
        name: &str,
        kernel: &Bound<'_, PyArray2<f32>>  // Принимаем numpy array
    ) -> PyResult<Self> {
        // Конвертируем numpy array в ndarray
        let array= unsafe {
            kernel.as_array().to_owned()
        };
        
        let conv = self.convolution_impl(kernel_size, Some((name, array)));
        Ok(Self { image: conv })
    }

    pub fn harris_corner_detection(&self, k: f32, threshold: f32, window_size: usize, sigma: f32) -> PyResult<Vec<(u32, u32)>>{
        let result = self.harris_corner_detection_impl(k, threshold, window_size, sigma);
        Ok(result.into_iter().collect())
    }

    pub fn sobel_operator(&self) -> PyResult<Self>{
        let sobel_result = self.sobel_operator_impl();
        Ok(Self{image: sobel_result})
    }

    pub fn canny_operator(&self, minthres: f32, maxthres: f32) -> PyResult<Self>{
        let canny_result = self.canny_operator_impl(minthres, maxthres);
        Ok(Self { image: canny_result })
    }

    pub fn fast(&self, threshold: f32, n: usize) -> PyResult<Vec<(u32, u32)>>{
        let fast_result = self.fast_impl(threshold, n);
        Ok(fast_result)
    }


}


impl PgmProcessor {
    fn mean_filter_impl(&self, kernel_size: usize) -> GrayImage {
        let (width, height) = self.image.dimensions();
        let mut output = ImageBuffer::new(width, height);
        let half_kernel = (kernel_size / 2) as i32;

        for x in 0..width {
            for y in 0..height {
                let mut sum = 0u32;
                let mut count = 0u32;

                for i in -half_kernel..=half_kernel {
                    for j in -half_kernel..=half_kernel {
                        let nx = x as i32 + i;
                        let ny = y as i32 + j;

                        if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                            let pixel = self.image.get_pixel(nx as u32, ny as u32);
                            sum += pixel[0] as u32;
                            count += 1;
                        }
                    }
                }

                let mean = (sum / count) as u8;
                output.put_pixel(x, y, Luma([mean]));
            }
        }
        output
    }

    fn median_filter_impl(&self, kernel_size: usize) -> GrayImage {
        let (width, height) = self.image.dimensions();
        let mut output = ImageBuffer::new(width, height);
        let half_kernel = (kernel_size / 2) as i32;

        for x in 0..width {
            for y in 0..height {
                let mut neighbors = Vec::new();

                for i in -half_kernel..=half_kernel {
                    for j in -half_kernel..=half_kernel {
                        let nx = x as i32 + i;
                        let ny = y as i32 + j;

                        if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                            let pixel = self.image.get_pixel(nx as u32, ny as u32);
                            neighbors.push(pixel[0]);
                        }
                    }
                }

                neighbors.sort();
                let median = neighbors[neighbors.len() / 2];
                output.put_pixel(x, y, Luma([median]));
            }
        }
        output
    }

    fn create_gauss_kernel_impl1(sigma: f32, kernel_size: usize) -> Array2<f32> {
        let mut rng = rand::rng();
        let mut kernel = Array2::zeros((kernel_size, kernel_size));
        let half_size = (kernel_size / 2) as i32;
        let mut sum = 0.0;
        let normal = Normal::new(0.0, sigma).unwrap();
        for i in 0..kernel_size {
            for j in 0..kernel_size {
                let x = i as i32 - half_size;
                let y = j as i32 - half_size;
                let value = normal.sample(&mut rng);
                kernel[[i, j]] = value;
                sum += value;
            }
        }

        // Нормализация
        for i in 0..kernel_size {
            for j in 0..kernel_size {
                kernel[[i, j]] /= sum;
            }
        }

        kernel
    }

    fn gaussian_filter_impl(&self, sigma: f32, kernel_size: usize) -> GrayImage {
        let kernel = Self::create_gauss_kernel_impl1(sigma, kernel_size);
        self.apply_kernel_impl(&kernel)
    }

    fn sigma_filter_impl(&self, sigma_threshold: u8, kernel_size: usize) -> GrayImage {
        let (width, height) = self.image.dimensions();
        let mut output = ImageBuffer::new(width, height);
        let half_kernel = (kernel_size / 2) as i32;

        for x in 0..width {
            for y in 0..height {
                let center_pixel = self.image.get_pixel(x, y)[0];
                let mut sum = 0u32;
                let mut count = 0u32;

                for i in -half_kernel..=half_kernel {
                    for j in -half_kernel..=half_kernel {
                        let nx = x as i32 + i;
                        let ny = y as i32 + j;

                        if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                            let pixel_value = self.image.get_pixel(nx as u32, ny as u32)[0];
                            
                            if pixel_value.abs_diff(center_pixel) <= sigma_threshold {
                                sum += pixel_value as u32;
                                count += 1;
                            }
                        }
                    }
                }

                if count > 0 {
                    let mean = (sum / count) as u8;
                    output.put_pixel(x, y, Luma([mean]));
                } else {
                    output.put_pixel(x, y, Luma([center_pixel]));
                }
            }
        }
        output
    }

    fn absolute_difference_map_impl(&self, filtered_image: &GrayImage) -> GrayImage {
        let (width, height) = self.image.dimensions();
        let mut diff_map = ImageBuffer::new(width, height);

        for x in 0..width {
            for y in 0..height {
                let original_pixel = self.image.get_pixel(x, y)[0];
                let filtered_pixel = filtered_image.get_pixel(x, y)[0];
                
                let diff = original_pixel.abs_diff(filtered_pixel);
                let enhanced_diff = diff.saturating_mul(5).min(255);
                
                diff_map.put_pixel(x, y, Luma([enhanced_diff]));
            }
        }
        diff_map
    }

    fn apply_kernel_impl(&self, kernel: &Array2<f32>) -> GrayImage {
        let (width, height) = self.image.dimensions();
        let kernel_size = kernel.shape()[0];
        let half_kernel = (kernel_size / 2) as i32;
        let mut output = ImageBuffer::new(width, height);

        for x in 0..width {
            for y in 0..height {
                let mut sum = 0.0f32;

                for i in 0..kernel_size {
                    for j in 0..kernel_size {
                        let nx = x as i32 + i as i32 - half_kernel;
                        let ny = y as i32 + j as i32 - half_kernel;

                        if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                            let pixel = self.image.get_pixel(nx as u32, ny as u32);
                            let weight = kernel[[i, j]];
                            
                            sum += pixel[0] as f32 * weight;
                        }
                    }
                }

                let value = sum.clamp(0.0, 255.0) as u8;
                output.put_pixel(x, y, Luma([value]));
            }
        }
        output
    }

    fn add_gaussian_noise_impl(&self, sigma: f32) -> GrayImage {
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, sigma).unwrap();
        let mut noisy_image = self.image.clone();

        for pixel in noisy_image.pixels_mut() {
            let noise = normal.sample(&mut rng);
            let new_value = (pixel[0] as f32 + noise).clamp(0.0, 255.0) as u8;
            pixel[0] = new_value;
        }
        noisy_image
    }

    fn add_salt_pepper_noise_impl(&self, salt_prob: f32, pepper_prob: f32) -> GrayImage {
        let mut rng = rand::rng();
        let mut noisy_image = self.image.clone();

        for pixel in noisy_image.pixels_mut() {
            let rand_val: f32 = rng.random();
            
            if rand_val < salt_prob {
                pixel[0] = 255;
            } else if rand_val < salt_prob + pepper_prob {
                pixel[0] = 0;
            }
        }
        noisy_image
    }

    fn unsharp_mask_impl(&self, blurred: &GrayImage, lambda: f32) -> GrayImage {
        let (width, height) = self.image.dimensions();
        let mut output = ImageBuffer::new(width, height);

        for x in 0..width {
            for y in 0..height {
                let original = self.image.get_pixel(x, y)[0] as f32;
                let blur_val = blurred.get_pixel(x, y)[0] as f32;

                // Формула: I + λ * (I - I_blur) Чтобы не забыть...
                let sharpened = original + lambda * (original - blur_val);

                
                let clamped = sharpened.clamp(0.0, 255.0) as u8;
                output.put_pixel(x, y, Luma([clamped]));
            }
        }
        output
    }
    //3 PART!

    pub fn high_pass_filter_impl(&self, coeff: f32, sigma: f32, filter: Option<&str>) -> GrayImage {
        let (width, height) = self.image.dimensions();
        let mut high_pass_image = GrayImage::new(width, height);
        
        match filter {
            Some("mean") => {
                let blurry_image = self.mean_filter_impl(3);
                let mut min_val = f32::MAX;
                let mut max_val = f32::MIN;
                
                for x in 0..width {
                    for y in 0..height {
                        let pixel_blur = blurry_image.get_pixel(x, y)[0] as f32;
                        let pixel_orig = self.image.get_pixel(x, y)[0] as f32;
                        let high_pass_pixel = pixel_orig - pixel_blur * coeff;
                        
                        min_val = min_val.min(high_pass_pixel);
                        max_val = max_val.max(high_pass_pixel);
                    }
                }
                
                let range = max_val - min_val;
                for x in 0..width {
                    for y in 0..height {
                        let pixel_blur = blurry_image.get_pixel(x, y)[0] as f32;
                        let pixel_orig = self.image.get_pixel(x, y)[0] as f32;
                        let mut high_pass_pixel = pixel_orig - pixel_blur * coeff;

                        if range > 0.0 {
                            high_pass_pixel = ((high_pass_pixel - min_val) / range) * 255.0;
                        }
                        
                        high_pass_image.put_pixel(x, y, Luma([high_pass_pixel.clamp(0.0, 255.0) as u8]));
                    }
                }
            },
            Some("gaussian") => {  
                let blurry_image = self.gaussian_filter_impl(sigma, 3);  // Используем гауссово размытие, а не шум!
                let mut min_val = f32::MAX;
                let mut max_val = f32::MIN;
                
                for x in 0..width {
                    for y in 0..height {
                        let pixel_blur = blurry_image.get_pixel(x, y)[0] as f32;
                        let pixel_orig = self.image.get_pixel(x, y)[0] as f32;
                        let high_pass_pixel = pixel_orig - pixel_blur * coeff;
                        
                        min_val = min_val.min(high_pass_pixel);
                        max_val = max_val.max(high_pass_pixel);
                    }
                }
                

                let range = max_val - min_val;
                for x in 0..width {
                    for y in 0..height {
                        let pixel_blur = blurry_image.get_pixel(x, y)[0] as f32;
                        let pixel_orig = self.image.get_pixel(x, y)[0] as f32;
                        let mut high_pass_pixel = pixel_orig - pixel_blur * coeff;
                        
                        // Нормализация
                        if range > 0.0 {
                            high_pass_pixel = ((high_pass_pixel - min_val) / range) * 255.0;
                        }
                        
                        high_pass_image.put_pixel(x, y, Luma([high_pass_pixel.clamp(0.0, 255.0) as u8]));
                    }
                }
            },
            None => {
                let blurry_image = self.gaussian_filter_impl(sigma, 3);  // По умолчанию используем гаусс
                let mut min_val = f32::MAX;
                let mut max_val = f32::MIN;
                

                for x in 0..width {
                    for y in 0..height {
                        let pixel_blur = blurry_image.get_pixel(x, y)[0] as f32;
                        let pixel_orig = self.image.get_pixel(x, y)[0] as f32;
                        let high_pass_pixel = pixel_orig - pixel_blur * coeff;
                        
                        min_val = min_val.min(high_pass_pixel);
                        max_val = max_val.max(high_pass_pixel);
                    }
                }
                
                let range = max_val - min_val;
                for x in 0..width {
                    for y in 0..height {
                        let pixel_blur = blurry_image.get_pixel(x, y)[0] as f32;
                        let pixel_orig = self.image.get_pixel(x, y)[0] as f32;
                        let mut high_pass_pixel = pixel_orig - pixel_blur * coeff;
                        
                        // Нормализация
                        if range > 0.0 {
                            high_pass_pixel = ((high_pass_pixel - min_val) / range) * 255.0;
                        }
                        
                        high_pass_image.put_pixel(x, y, Luma([high_pass_pixel.clamp(0.0, 255.0) as u8]));
                    }
                }
            }
            _ => {
                panic!("Нужно выбрать фильтр: 'mean' или 'gaussian'")
            }
        }
        
        high_pass_image
    }



    //не работает, выход зза границы, доработать!
    pub fn convolution_impl(&self, kernel_size: usize, kernel: Option<(&str, Array2<f32>)>) -> GrayImage {
        let (width, height) = self.image.dimensions();
        let window_half = (kernel_size / 2) as i32;

        let float_data = Array2::from_shape_fn((height as usize, width as usize), |(y, x)| {
            self.image.get_pixel(x as u32, y as u32)[0] as f32
        });

        let mut result = Array2::zeros((height as usize, width as usize));

        match kernel {
            Some(("median", _)) => {
                for y in window_half..(height as i32 - window_half) {
                    for x in window_half..(width as i32 - window_half) {
                        let mut neighbors = Vec::new();
                        for ky in -window_half..=window_half {
                            for kx in -window_half..=window_half {
                                let nx = x + kx;
                                let ny = y + ky;
                                if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                                    neighbors.push(float_data[(ny as usize, nx as usize)]);
                                }
                            }
                        }
                        neighbors.sort_by(f32::total_cmp);
                        result[(y as usize, x as usize)] = neighbors[neighbors.len() / 2];
                    }
                }
            }
            Some(("mean", _)) => {
                let kernel_weight = 1.0 / (kernel_size.pow(2)) as f32;
                for y in window_half..(height as i32 - window_half) {
                    for x in window_half..(width as i32 - window_half) {
                        let mut sum: f32 = 0.0;
                        for ky in -window_half..=window_half {
                            for kx in -window_half..=window_half {
                                let nx = x + kx;
                                let ny = y + ky;
                                if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                                    sum += float_data[(ny as usize, nx as usize)] * kernel_weight;
                                }
                            }
                        }
                        result[(y as usize, x as usize)] = sum;
                    }
                }
            }
            None => {
                return GrayImage::new(width, height);
            }
            Some((_, custom_kernel)) => {
                for y in window_half..(height as i32 - window_half) {
                    for x in window_half..(width as i32 - window_half) {
                        let mut sum: f32 = 0.0;
                        for ky in -window_half..=window_half {
                            for kx in -window_half..=window_half {
                                let nx = x + kx;
                                let ny = y + ky;
                                if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                                    let kernel_val = custom_kernel[((ky + window_half) as usize, (kx + window_half) as usize)];
                                    sum += float_data[(ny as usize, nx as usize)] * kernel_val;
                                }
                            }
                        }
                        result[(y as usize, x as usize)] = sum;
                    }
                }
            }
        }

        let buf: Vec<u8> = result.iter().map(|&x| x.clamp(0.0, 255.0) as u8).collect();
        let image = GrayImage::from_vec(width, height, buf).unwrap();
        image
    }

    pub fn harris_corner_detection_impl(&self, k: f32, threshold: f32, window_size: usize, sigma: f32) -> Array1<(u32, u32)>{
        let (ix, iy) = self.compute_gradients();
        
        let ix2 = self.elem_multiply(&ix, &ix);
        let iy2 = self.elem_multiply(&iy, &iy);
        let ixy = self.elem_multiply(&ix, &iy);

        let ix2_smooth = self.gauss_blur_for_array(&ix2, window_size as i32, sigma);
        let iy2_smooth = self.gauss_blur_for_array(&iy2, window_size as i32, sigma);
        let ixy_smooth = self.gauss_blur_for_array(&ixy, window_size as i32, sigma);
        

        let response = self.compute_harris_response(&ix2_smooth, &iy2_smooth, &ixy_smooth, k);
        
        let corners = self.non_maximum_suppression(&response, threshold);
        
        corners
    }


    //На подумать, оставлять ли функцию или использовать другой подход...
    fn gauss_blur_for_array(&self, array: &Array2<f32>, window_size: i32, sigma: f32) -> Array2<f32> {
        let (width, height) = array.dim();
        let window_half = window_size / 2;
        let mut result = Array2::zeros((width, height));
        

        let kernel = self.create_gauss_kernel_impl(window_size, sigma);
        
        for y in window_half..(height as i32 - window_half) {
            for x in window_half..(width as i32 - window_half) {
                let mut sum = 0.0;
                // let mut weight_sum = 0.0;
                
                for ky in -window_half..=window_half {
                    for kx in -window_half..=window_half {
                        let kernel_val = kernel[[(ky + window_half) as usize, (kx + window_half) as usize]];
                        let pixel_val = array[[(x + kx) as usize, (y + ky) as usize]];
                        
                        sum += pixel_val * kernel_val;
                        // weight_sum += kernel_val;
                    }
                }
                
                result[[x as usize, y as usize]] = sum;
            }
        }
        
        result
    }

    fn create_gauss_kernel_impl(&self, size: i32, sigma: f32) -> Array2<f32> {
        let mut kernel = Array2::zeros((size as usize, size as usize));
        let half = size / 2;
        let sigma_sq = 2.0 * sigma * sigma;
        let mut sum = 0.0;
        

        for y in -half..=half {
            for x in -half..=half {
                let distance_sq = (x * x + y * y) as f32;
                let value = (-distance_sq / sigma_sq).exp();
                
                kernel[[(y + half) as usize, (x + half) as usize]] = value;
                sum += value;
            }
        }
        
        for y in 0..size as usize {
            for x in 0..size as usize {
                kernel[[y, x]] /= sum;
            }
        }
        
        kernel
    }

    fn compute_gradients(&self) -> (Array2<f32>, Array2<f32>) {
        let (width, height) = self.image.dimensions();
        let mut ix = Array2::<f32>::zeros((width as usize, height as usize));
        let mut iy = Array2::<f32>::zeros((width as usize, height as usize));

        // Ядра Собеля
        let sobel_x = [[-1.0, 0.0, 1.0],
                    [-2.0, 0.0, 2.0],
                    [-1.0, 0.0, 1.0]];
        let sobel_y = [[-1.0, -2.0, -1.0],
                    [ 0.0,  0.0,  0.0],
                    [ 1.0,  2.0,  1.0]];

        for x in 1..(width - 1) {
            for y in 1..(height - 1) {
                let mut gx = 0.0;
                let mut gy = 0.0;

                for i in 0..3 {
                    for j in 0..3 {
                        let img_x = (x as i32 + i as i32 - 1) as u32;
                        let img_y = (y as i32 + j as i32 - 1) as u32;
                        let pixel = self.image.get_pixel(img_x, img_y)[0] as f32;
                        gx += pixel * sobel_x[i][j];
                        gy += pixel * sobel_y[i][j];
                    }
                }

                ix[[x as usize, y as usize]] = gx;
                iy[[x as usize, y as usize]] = gy;
            }
        }

        (ix, iy)
    }

    fn elem_multiply(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32>{
        let (width, height) = (a.dim().0, a.dim().1);
        let mut result = Array2::<f32>::zeros((width, height));

        for x in 0..width{
            for y in 0..height{
                result[[x, y]] = a[[x, y]] * b[[x, y]];
            }
        }
        return result;
    }

    fn compute_harris_response(&self, ix2: &Array2<f32>, iy2: &Array2<f32>, ixy: &Array2<f32>, k: f32) -> Array2<f32>{
        let (width, height) = (ix2.dim().0, ix2.dim().1);

        let mut response = Array2::<f32>::zeros((width, height));

        for x in 0..width{
            for y in 0..height{

                //det ix2 * iy2 - ixy^2
                let det = ix2[[x, y]] * iy2[[x, y]] - ixy[[x, y]].powi(2);
                
                //ix2 + iy2
                let trace = ix2[[x, y]] + iy2[[x, y]];

                //response = det - k * trce^2
                response[[x, y]] = det - k * trace * trace; 

            }
        }
        
        
        return response;
    }

    fn non_maximum_suppression(&self, response: &Array2<f32>, threshold: f32) -> Array1<(u32, u32)>{
        let (width, height) = (response.dim().0, response.dim().1);

        let mut corners = Vec::new();
        let window_size = 3;

        for x in window_size..width - window_size {
            for y in window_size..height - window_size {
                let current_response = response[[x, y]];
                
                if current_response < threshold {
                    continue;
                }


                let mut is_local_max = true;
                for i in -1..=1 {
                    for j in -1..=1 {
                        if i == 0 && j == 0 {
                            continue;
                        }
                        let nx = (x as i32 + i) as usize;
                        let ny = (y as i32 + j) as usize;
                        if response[[nx, ny]] >= current_response {
                            is_local_max = false;
                            break;
                        }
                    }
                    if !is_local_max {
                        break;
                    }
                }
                
                if is_local_max {
                    corners.push((x as u32, y as u32));
                }
            }
        }
        
        let array_corners = Array1::<(u32, u32)>::from(corners);
        return array_corners;

    }


    pub fn sobel_operator_impl(&self) -> GrayImage {
        let (width, height) = self.image.dimensions();
        let mut edge_image = GrayImage::new(width, height);

        let sobel_x = Array2::from_shape_vec((3, 3), vec![
            -1.0, 0.0, 1.0,
            -2.0, 0.0, 2.0,
            -1.0, 0.0, 1.0,
        ]).unwrap();

        let sobel_y = Array2::from_shape_vec((3, 3), vec![
            -1.0, -2.0, -1.0,
            0.0,  0.0,  0.0,
            1.0,  2.0,  1.0,
        ]).unwrap();

        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                let mut pixel_x: f32 = 0.0;
                let mut pixel_y: f32 = 0.0;

                for i in -1..=1 {
                    for j in -1..=1 {
                        let px = (x as i32 + i) as u32;
                        let py = (y as i32 + j) as u32;
                        let pixel = self.image.get_pixel(px, py)[0] as f32;

                        pixel_x += pixel * sobel_x[[(i + 1) as usize, (j + 1) as usize]];
                        pixel_y += pixel * sobel_y[[(i + 1) as usize, (j + 1) as usize]];
                    }
                }

                let magnitude = (pixel_x * pixel_x + pixel_y * pixel_y).sqrt();
                let norm_magnitude = magnitude.min(255.0) as u8;
                edge_image.put_pixel(x, y, Luma([norm_magnitude]));
            }
        }

        edge_image
    }

    pub fn canny_operator_impl(&self, minthres: f32, maxthres: f32) -> GrayImage {
        let (width, height) = self.image.dimensions();
        let blurry_image = self.gaussian_filter_impl(1.0, 5); // sigma=1.0, kernel=5 — более реалистично

        let mut magnitude_image = GrayImage::new(width, height);
        let mut directions = Array2::<f32>::zeros((width as usize, height as usize));
        let mut result_image = GrayImage::new(width, height);

        let sobel_x = Array2::<f32>::from_shape_vec((3, 3), vec![
            -1.0, 0.0, 1.0,
            -2.0, 0.0, 2.0,
            -1.0, 0.0, 1.0,
        ]).unwrap();

        let sobel_y = Array2::<f32>::from_shape_vec((3, 3), vec![
            -1.0, -2.0, -1.0,
            0.0,  0.0,  0.0,
            1.0,  2.0,  1.0,
        ]).unwrap();

        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                let mut pixel_x: f32 = 0.0;
                let mut pixel_y: f32 = 0.0;

                for i in -1..=1 {
                    for j in -1..=1 {
                        let px = (x as i32 + i) as u32;
                        let py = (y as i32 + j) as u32;
                        let pixel = blurry_image.get_pixel(px, py)[0] as f32;

                        pixel_x += pixel * sobel_x[[(i + 1) as usize, (j + 1) as usize]];
                        pixel_y += pixel * sobel_y[[(i + 1) as usize, (j + 1) as usize]];
                    }
                }

                let magnitude = (pixel_x * pixel_x + pixel_y * pixel_y).sqrt();
                let theta = pixel_y.atan2(pixel_x);

                magnitude_image.put_pixel(x, y, Luma([magnitude.min(255.0) as u8]));
                directions[[x as usize, y as usize]] = theta;
            }
        }

        let suppressed = self.non_maximum_suppresion_canny(magnitude_image, directions);
        self.hysteresis_threshold(minthres, maxthres, &mut result_image, &suppressed);

        result_image
    }


    fn non_maximum_suppresion_canny(&self, magnitude: GrayImage, directions: Array2<f32>) -> GrayImage{
        let (width, height) = self.image.dimensions();
        let mut suppresed = GrayImage::new(width, height);

        for y in 1..(height-1) as usize{
            for x in 1..(width - 1) as usize{
                let angle = directions[[x, y]];
                let mag = magnitude.get_pixel(x as u32, y as u32)[0] as f32;


                let (dx1, dx2, dy1, dy2) = match self.rounded_angle(angle){
                    0 => (1i32, 0, -1i32, 0),  // Горизонтальное направление
                    45 => (1, -1, -1, 1), // Диагональ 45°
                    90 => (0, -1, 0, 1),  // Вертикальное направление  
                    135 => (-1, -1, 1, 1),// Диагональ 135°
                    _ => (0, 0, 0, 0) 
                };

                let mag1 = magnitude.get_pixel((x as i32 + dx1) as u32, (y as i32 + dy1) as u32)[0] as f32;
                let mag2 = magnitude.get_pixel((x as i32 + dx2) as u32, (y as i32 + dy2) as u32)[0] as f32;

                if mag >= mag1 && mag >= mag2{
                    suppresed.put_pixel(x as u32, y as u32, Luma([mag.min(255.0) as u8]));
                }else {
                    suppresed.put_pixel( x as u32, y as u32, Luma([0]));
                }
            }
        }
        
        return suppresed;
    }

    fn rounded_angle(&self, angle: f32) -> i32{
        let angle_deg = angle.to_degrees();
        let positive_angle = if angle_deg < 0.0 { angle_deg + 180.0 } else { angle_deg };
    
        if (positive_angle >= 0.0 && positive_angle < 22.5) || (positive_angle >= 157.5 && positive_angle <= 180.0) {
            return 0;   // Горизонтальное
        } else if positive_angle >= 22.5 && positive_angle < 67.5 {
            return 45;  // Диагональ 45°
        } else if positive_angle >= 67.5 && positive_angle < 112.5 {
            return 90;  // Вертикальное
        } else {
            return 135; // Диагональ 135°
        }
    }

    fn hysteresis_threshold(&self, min_thres: f32, max_thres: f32, result: &mut GrayImage, suppresed: &GrayImage){
        let (width, height) = suppresed.dimensions();
        
        let mut visited = Array2::from_elem((width as usize, height as usize), false);
        let mut strong_edges = Vec::new();

        for y in 0..height{
            for x in 0..width{
                let mag = suppresed.get_pixel(x , y)[0] as f32;

                if mag >= max_thres{
                    result.put_pixel(x, y, Luma([255]));
                    strong_edges.push((x, y));
                    visited[[x as usize, y as usize]] = true;
                }
                else if mag >= min_thres {
                    result.put_pixel(x, y, Luma([0]));
                }
                else{
                    result.put_pixel(x, y, Luma([0]));
                }
            }

        }

        while let Some((x, y)) = strong_edges.pop(){
            for i in -1..=1 as i32{
                for j in -1..=1 as i32{
                    if i == 0 && j == 0{
                        continue;
                    }

                    let nx = x as i32 + i;
                    let ny = y as i32 + j;

                    if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32{
                        if !visited[[nx as usize, ny as usize]]{

                            let neighbor_mag = suppresed.get_pixel(x, y)[0] as f32;
                            if neighbor_mag >= min_thres{
                                result.put_pixel(x, y, Luma([255]));
                                strong_edges.push((nx as u32, ny as u32));
                                visited[[nx as usize, ny as usize]] = true;
                            }
                        }
                    }
                }

            }

        }
    }

    pub fn fast_impl(&self, threshold: f32, n: usize) -> Vec<(u32, u32)>{
        let (width, height) = self.image.dimensions();
        let mut corners = Vec::new();
        let circle = [
            (0, -3), (1, -3), (2, -2), (3, -1), (3, 0), (3, 1), (2, 2), (1, 3),
            (0, 3), (-1, 3), (-2, 2), (-3, 1), (-3, 0), (-3, -1), (-2, -2), (-1, -3)
        ];

        for y in 3..(height - 3){
            for x in 3..(width - 3){
                let intensive = self.image.get_pixel(x, y)[0] as f32;

                if !self.fast_check(x, y, intensive, threshold, &circle){
                    continue;
                };


                if self.full_circle_check(x, y, intensive, threshold, n, &circle){
                    corners.push((x, y));
                }

            }
        }

        return corners;

    }

    fn fast_check(&self, x: u32, y: u32, intensive: f32, threshold: f32, circle: &[(i32, i32)]) -> bool {
        let (width, height) = self.image.dimensions();
        let mut bright_counter = 0;
        let mut dark_counter = 0;

        for &i in &[0, 4, 8, 12] {
            let (dx, dy) = circle[i];
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;

            if nx < 0 || nx >= width as i32 || ny < 0 || ny >= height as i32 {
                continue;
            }

            let neighbor_intensive = self.image.get_pixel(nx as u32, ny as u32)[0] as f32;

            if neighbor_intensive > intensive + threshold {
                bright_counter += 1;
            } else if neighbor_intensive < intensive - threshold {
                dark_counter += 1;
            }
        }

        bright_counter >= 3 || dark_counter >= 3
    }

    fn full_circle_check(&self, x: u32, y: u32, intensive: f32, threshold: f32, n: usize, circle: &[(i32, i32)]) -> bool{
        let mut types = [0; 16];

        for (i, &(dx, dy)) in circle.iter().enumerate(){
            let neighbor_intensive = self.image.get_pixel(
                (x as i32 + dx) as u32, (y as i32 + dy) as u32
            )[0] as f32;

            if neighbor_intensive > intensive + threshold{
                types[i] = 1; //1 для светлых пикселей сделаем, 0 для нейтральных 
            }
            else if neighbor_intensive < intensive - threshold{
                types[i] = 2 //сделаем для темных пикселей
            }
        }


        return self.find_continuos_segment(&types, n);
    }

    fn find_continuos_segment(&self, types: &[i32; 16], n: usize) -> bool{
        for start in 0..16{
            if types[start] == 0{
                continue;
            }

            let mut lenght = 0;
            for offset in 0..16{
                let idx = (start + offset) % 16;
                if types[idx] == types[start]{
                    lenght += 1;
                    if lenght == n{
                        return true;
                    }
                }else {
                    break;
                }
            }
        }
        return false;
    }

}

#[pymodule]
fn pgm_processor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PgmProcessor>()?;
    Ok(())
}
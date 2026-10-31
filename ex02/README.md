# variances as-is

The distinct magnitude of the features shows that the variance concentrates i features of large magnitude. You learn that  standardization or normalization must be applied first.

+ ![Variances without normalization](variances_plot_0.png)

# Variances and Standardization
Standardization implies that all features have mean of approximately zero and a standard deviation of one, resulting in a variance of unity. 
 
+ ![Variances with Stadarization](variances_plot_1.png)



# Variances and normalization (Min-Max)
Min-Max scaling transforms all valies into the range zero to one. While this preserves the original variance of each individual feature, the realtive order of magnitude among the variances of all features remains unchanged. COnsequently, each varaicne is confined to the same scale.  


+ ![Variances with Normalization (Min-Max) including "Knight"](variances_plot_2.png)

Given that the variance of the target feature is one, I must exclude it to present a chart demonstrating how each individual feature explains the total variance.

+ ![Variances with Normalization (Min-Max) without "Knight"](variances_plot_3.png)


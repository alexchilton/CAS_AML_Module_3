# SOLUTION
# using trained AE model obtain reconstruction:
test_images_rec = dae.predict(test_images)
residuals = (test_images - test_images_rec)

mse = (residuals**2).mean(axis=(1, 2))
mse_cutoff =  mse.mean() + mse.std()*3

plt.hist(mse, 100);
plt.axvline(mse_cutoff, c='r')
plt.xlabel('mse')
plt.show()
plt.close()

outliers_mask = mse > mse_cutoff

n_outliers = sum(outliers_mask)
print(f'numer of outliers: {n_outliers} out of {len(test_images)} samples')

outliers = test_images[outliers_mask]
non_outliers = test_images[~outliers_mask]

outliers_rec = test_images_rec[outliers_mask]
non_outliers_rec = test_images_rec[~outliers_mask]

# make square canvas
# get square side
n_side = int(np.ceil(np.sqrt(n_outliers)))

# pad n_outliers array with black squares till total size of n_side**2
padding = np.zeros([n_side**2 - n_outliers, 28, 28])

outliers_padded = np.concatenate([outliers, padding])
outliers_rec_padded = np.concatenate([outliers_rec, padding])

# create canvas with the mosaic functions
outliers_mosaic = mosaic(outliers_padded.reshape((n_side, n_side, 28, 28)))
non_outliers_mosaic = mosaic(non_outliers[:n_side**2].reshape((n_side, n_side, 28, 28)))

outliers_rec_mosaic = mosaic(outliers_rec_padded.reshape((n_side, n_side, 28, 28)))
non_outliers_rec_mosaic = mosaic(non_outliers_rec[:n_side**2].reshape((n_side, n_side, 28, 28)))

plt.imshow(outliers_mosaic, cmap='gray')
plt.axis('off')
plt.title('outliers')
plt.show()
plt.close()

plt.imshow(outliers_rec_mosaic, cmap='gray')
plt.axis('off')
plt.title('outliers reconstruction')
plt.show()
plt.close()

plt.imshow(non_outliers_mosaic, cmap='gray')
plt.axis('off')
plt.title('non-outliers')
plt.show()
plt.close()

plt.imshow(non_outliers_rec_mosaic, cmap='gray')
plt.axis('off')
plt.title('non-outliers reconstruction')
plt.show()
plt.close()
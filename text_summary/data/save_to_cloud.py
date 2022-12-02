from google.cloud import storage

def save_cloud(file):
    '''
    Saves data to cloud
    '''
    images = get_images()
    # Save images locally first
    save_images_local(images)
    # Then transfer to cloud
    print('Saving images to cloud')
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    for i, filename in enumerate(images.keys()):
        if verbose:
            print(f'Saving image {i}')
        cloud_filename = f"data-images/{filename}"
        machine_filename = os.path.join(get_image_path(), filename)
        blob = bucket.blob(cloud_filename)
        blob.upload_from_filename(machine_filename)
        # Delete local data after uploading to cloud
        if os.path.exists(machine_filename):
            os.remove(machine_filename)

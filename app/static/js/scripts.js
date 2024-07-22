document.getElementById('upload-form').addEventListener('submit', async function (event) {
    event.preventDefault();
    const outputDiv = document.getElementById('output');
    const imageInput = document.getElementById('imageInput');
    const modelInput = document.getElementById('modelInput');
    const originalImage = document.getElementById('originalImage');
    const colorizedImage = document.getElementById('colorizedImage');

    if (imageInput.files.length === 0) {
        alert('Please select an image file.');
        return;
    }

    const file = imageInput.files[0];
    const formData = new FormData();
    formData.append('file', file);
    formData.append('modelId', modelInput.value);

    // Display the original image
    const reader = new FileReader();
    reader.onload = function (e) {
        originalImage.src = e.target.result;
    }
    reader.readAsDataURL(file);

    colorizedImage.src = "";
    outputDiv.setAttribute("class", "mt-4 row");

    try {
        const response = await fetch('/colorize/', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const body = await response.json();
            throw new Error(`Failed to colorize image: ${body.error}`);
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        colorizedImage.src = url;
    } catch (error) {
        console.error('Error:', error);
        alert('Error colorizing image. Please try again.');
    }
});

// Animation on Scroll
const hiddenElement1 = document.querySelectorAll('.hidden');
const hiddenElement2 = document.querySelectorAll('.hidden-left');
const hiddenElement3 = document.querySelectorAll('.hidden-right');
const hiddenElement4 = document.querySelectorAll('.fade-in');


//Intersection Observer that will handle the animation
const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    console.log(entry)
    if (entry.isIntersecting) {
      entry.target.classList.add('show');
    } else {
      entry.target.classList.remove('show');
    }
  });
});

// Observe the elements
hiddenElement1.forEach((el) => observer.observe(el));
hiddenElement2.forEach((el) => observer.observe(el));
hiddenElement3.forEach((el) => observer.observe(el));
hiddenElement4.forEach((el) => observer.observe(el));


// Scroll to page button Function
$(document).ready(function () {
  // Handle button click event
  $('#button-p').on('click', function () {
    // Scroll to the 'detect' section
    $('html, body').animate({
      scrollTop: $('#detect').offset().top
    }, 500); //
  });
});


// Navigation Bar Function
$(document).ready(function () {
  // Cache the navigation links and sections
  var navLinks = $('.menu-text');
  var sections = $('section');

  $(window).on('scroll', function () {
    // Get the current scroll position
    var scrollPosition = $(this).scrollTop();

    // Check each section's position relative to the viewport
    sections.each(function () {
      var sectionTop = $(this).offset().top;
      var sectionBottom = sectionTop + $(this).outerHeight();

      // If the current scroll position is within the section
      if (scrollPosition >= sectionTop && scrollPosition < sectionBottom) {
        // Remove present class from all navigation items
        navLinks.removeClass('present');

        // Find the corresponding navigation item and add present class
        var targetId = $(this).attr('id');
        $('[href="#' + targetId + '"]').addClass('present');

        // Break the loop since we found the present section
        return false;
      }
    });
  });
});


// Display Image on Div
function displayImagePreview() {
  // Get the uploaded image file
  const imageFile = document.getElementById('upload').files[0];

  // Check if an image file was uploaded
  if (imageFile) {
    // Create an object URL for the image file
    const imageURL = URL.createObjectURL(imageFile);

    // Set the div's content to the image URL
    document.querySelector('.img-container').innerHTML = `
        <img src="${imageURL}" alt="Uploaded Image" width="224" height="224">
      `;
  }
}


// Enable Loader
const loader = document.querySelector("#wifi-loader");
window.onload = function () {
  const submitbtn = document.querySelector(".submit");
  submitbtn.addEventListener("click", () => {
    // Show the loader
    loader.style.display = 'flex';
    console.log('true');
  });
}


// Form Function/Detection
const uploadInput = document.getElementById('upload');
uploadInput.addEventListener('change', function () {
  // Get the uploaded image file
  let imageFile = this.files[0];

  // Check if an image file was uploaded
  if (imageFile) {
    // Check if the file is a .PNG or .JPG file
    if (imageFile.type !== 'image/png' && imageFile.type !== 'image/jpeg') {
      // Clear the input field
      this.value = '';

      // Display an alert
      alert('Only .PNG and .JPG files are accepted.');
    } else {
      // Display the image preview
      displayImagePreview();
    }
  }
});

// const uploadInput = document.getElementById('upload');
uploadInput.addEventListener('change', displayImagePreview);

document.querySelector('.form-container').addEventListener('submit', function (event) {
  // Prevent the form from being submitted normally
  event.preventDefault();

  // Clear the output div
  document.querySelector('.output').innerHTML = '';

  // Create a new FormData object from the form
  var formData = new FormData(event.target);

  // Send the form data with Fetch
  fetch('/predict', {
    method: 'POST',
    body: formData
  })
    .then(response => response.json())
    .then(data => {
      // Update the page with the response data
      var predicted_class = data.predicted_class;
      var output_text = '';
      if (predicted_class == 0) {
        output_text = '<a class="output-text" href="#blb">Bacterial Leaf Blight</a>';
        loader.style.display = 'none';
      }
      else if (predicted_class == 1) {
        output_text = '<a class="output-text" href="#bls">Bacterial Leaf Streak</a>';
        loader.style.display = 'none';
      }
      else if (predicted_class == 2) {
        output_text = '<a class="output-text" href="#bakanae">Bakanae</a>';
        loader.style.display = 'none';
      }
      else if (predicted_class == 3) {
        output_text = '<a class="output-text" href="#bs">Brown Spot</a>';
        loader.style.display = 'none';
      }
      else if (predicted_class == 4) {
        output_text = '<a class="output-text" href="#gsv">Rice Grassy Stunt Virus</a>';
        loader.style.display = 'none';
      }
      else if (predicted_class == 5) {
        output_text = '<a class="output-text" href="#healthy-rice">Healthy Rice Plant</a>';
        loader.style.display = 'none';
      }
      else if (predicted_class == 6) {
        output_text = '<a class="output-text" href="#nbs">Narrow Brown Spot</a>';
        loader.style.display = 'none';
      }
      else if (predicted_class == 7) {
        output_text = '<a class="output-text" href="#nr">Not Rice</a>';
        loader.style.display = 'none';
      }
      else if (predicted_class == 8) {
        output_text = '<a class="output-text" href="#rsv">Ragged Stunt Virus</a>';
        loader.style.display = 'none';
      }
      else if (predicted_class == 9) {
        output_text = '<a class="output-text" href="#rb">Rice Blast</a>';
        loader.style.display = 'none';
      }
      else if (predicted_class == 10) {
        output_text = '<a class="output-text" href="#rfs">Rice False Smut</a>';
        loader.style.display = 'none';
      }
      else if (predicted_class == 11) {
        output_text = '<a class="output-text" href="#sb">Sheath Blight</a>';
        loader.style.display = 'none';
      }
      else if (predicted_class == 12) {
        output_text = '<a class="output-text" href="#sr">Sheath Rot</a>';
        loader.style.display = 'none';
      }
      else if (predicted_class == 13) {
        output_text = '<a class="output-text" href="#str">Stem Rot</a>';
        loader.style.display = 'none';
      }
      else if (predicted_class == 14) {
        output_text = '<a class="output-text" href="#tv">Tungro Virus</a>';
        loader.style.display = 'none';
      }
      else {
        output_text = '<a class="output-text no-line">No file uploaded</a>';
        loader.style.display = 'none';
      }
      document.querySelector('.output').innerHTML = output_text;
    })
    .catch(error => {
      console.error('Error:', error);
    });
});


// reset
document.querySelector('.reset').addEventListener('click', function (event) {
  // Prevent the button from submitting the form
  event.preventDefault();

  // Clear the file input
  document.querySelector('.upload-image').value = '';

  // Reset the image preview
  document.querySelector('.img-container').innerHTML = '<img src="/static/svg/no image.svg" alt="🌿"></img>';

  // Clear the output div
  document.querySelector('.output').innerHTML = '';

  // set loader to none
  loader.style.display = 'none';
});

// JS functions specific for mobile
// Hamburger menu clickevents
const hamburger = document.getElementById('hamburger');
const navigation_menu = document.getElementById('menu-nav');
const navigation_text = document.querySelectorAll('.menu-item');

hamburger.addEventListener('click', () => {
  hamburger.classList.toggle('active');
  document.body.classList.toggle('active');
  navigation_menu.classList.toggle('active');

  // Check if the menu is being opened or closed
  if (navigation_menu.classList.contains('active')) {
    // If the menu is being opened, add the 'entrance' class
    navigation_text.forEach(text => {
      text.classList.add('entrance');
    });
  } else {
    // If the menu is being closed, remove the 'entrance' class
    navigation_text.forEach(text => {
      text.classList.remove('entrance');
    });
  }
});


// Add event listener to each menu item
navigation_text.forEach(function(menu_item) {
  menu_item.addEventListener('click', function() {
    // Hide the menu and change the state of the hamburger button
    document.body.classList.remove('active');
    hamburger.classList.remove('active');
    navigation_menu.classList.remove('active');

    // Remove the 'entrance' class from each navigation text
    navigation_text.forEach(text => {
      text.classList.remove('entrance');
    });
  });
});


// Add event listener to arrow
const arrows = document.querySelectorAll('.arrow-container');
const arrow = document.querySelectorAll('.arrow')

arrows.forEach((arrowContainer, index) => {
  arrowContainer.addEventListener('click', () => {
    // Get the parent .sd-content of the clicked arrow
    const parentContent = arrowContainer.closest('.sd-content');
    // Toggle the 'clicked' class for the parent .sd-content
    parentContent.classList.toggle('clicked');
    // Toggle the 'flip' class for the corresponding arrow
    arrow[index].classList.toggle('flip');
  })
})

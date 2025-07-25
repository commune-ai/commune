// Toggle sidebar functionality for logo click
document.addEventListener('DOMContentLoaded', () => {
  // Function to toggle right sidebar
  const toggleRightSidebar = () => {
    const event = new CustomEvent('toggleRightSidebar');
    window.dispatchEvent(event);
  };

  // Add click handler to logo
  const addLogoClickHandler = () => {
    const logoLink = document.querySelector('a[href="/"]');
    if (logoLink) {
      logoLink.addEventListener('click', (e) => {
        // Check if we're already on the home page
        if (window.location.pathname === '/') {
          e.preventDefault();
          toggleRightSidebar();
        }
      });
    }
  };

  // Try to add handler immediately and also observe for dynamic content
  addLogoClickHandler();
  
  // Observer for when header is dynamically loaded
  const observer = new MutationObserver(() => {
    addLogoClickHandler();
  });
  
  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
});

// Simple pagination implementation
const ITEMS_PER_PAGE = 5;
let currentPage = 1;
let totalItems = 0;
let allData = [];

// Sample data generator
function generateSampleData() {
    const data = [];
    for (let i = 1; i <= 23; i++) {
        data.push({
            id: i,
            title: `Item ${i}`,
            description: `This is the description for item number ${i}. It contains some sample text to demonstrate the pagination functionality.`
        });
    }
    return data;
}

// Initialize data
allData = generateSampleData();
totalItems = allData.length;

// Calculate total pages
function getTotalPages() {
    return Math.ceil(totalItems / ITEMS_PER_PAGE);
}

// Get items for current page
function getPageItems(page) {
    const startIndex = (page - 1) * ITEMS_PER_PAGE;
    const endIndex = startIndex + ITEMS_PER_PAGE;
    return allData.slice(startIndex, endIndex);
}

// Render content
function renderContent() {
    const container = document.getElementById('content-container');
    const items = getPageItems(currentPage);
    
    container.innerHTML = '';
    
    if (items.length === 0) {
        container.innerHTML = '<p>No items to display.</p>';
        return;
    }
    
    items.forEach(item => {
        const itemElement = document.createElement('div');
        itemElement.className = 'content-item';
        itemElement.innerHTML = `
            <h3>${item.title}</h3>
            <p>${item.description}</p>
        `;
        container.appendChild(itemElement);
    });
}

// Update pagination UI
function updatePaginationUI() {
    const totalPages = getTotalPages();
    const pageInfo = document.getElementById('page-info');
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    
    // Update page info
    pageInfo.textContent = `Page ${currentPage} of ${totalPages}`;
    
    // Update button states
    prevBtn.disabled = currentPage === 1;
    nextBtn.disabled = currentPage === totalPages;
}

// Handle page navigation
function goToPage(page) {
    const totalPages = getTotalPages();
    if (page < 1 || page > totalPages) return;
    
    currentPage = page;
    renderContent();
    updatePaginationUI();
}

// Event listeners
document.getElementById('prev-btn').addEventListener('click', () => {
    goToPage(currentPage - 1);
});

document.getElementById('next-btn').addEventListener('click', () => {
    goToPage(currentPage + 1);
});

// Initialize the page
document.addEventListener('DOMContentLoaded', () => {
    renderContent();
    updatePaginationUI();
});

// Optional: Add keyboard navigation
document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') {
        goToPage(currentPage - 1);
    } else if (e.key === 'ArrowRight') {
        goToPage(currentPage + 1);
    }
});
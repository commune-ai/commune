// Dope Vibe Generator JavaScript

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const vibeTypeSelect = document.getElementById('vibe-type');
    const intensitySlider = document.getElementById('intensity');
    const intensityValue = document.getElementById('intensity-value');
    const generateBtn = document.getElementById('generate-btn');
    const playBtn = document.getElementById('play-btn');
    const pauseBtn = document.getElementById('pause-btn');
    const saveBtn = document.getElementById('save-btn');
    const savedVibesList = document.getElementById('saved-vibes-list');
    
    // Vibe display elements
    const currentVibeType = document.getElementById('current-vibe-type');
    const vibeBpm = document.getElementById('vibe-bpm');
    const vibeKey = document.getElementById('vibe-key');
    const vibeScale = document.getElementById('vibe-scale');
    const vibeInstruments = document.getElementById('vibe-instruments');
    const vibeEffects = document.getElementById('vibe-effects');
    
    // Canvas for visualization
    const canvas = document.getElementById('visualizer');
    const ctx = canvas.getContext('2d');
    
    // Resize canvas to fit container
    function resizeCanvas() {
        const container = canvas.parentElement;
        canvas.width = container.offsetWidth;
        canvas.height = container.offsetHeight;
    }
    
    // Call resize on load and window resize
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    
    // Current vibe state
    let currentVibe = null;
    let isPlaying = false;
    let animationId = null;
    let savedVibes = JSON.parse(localStorage.getItem('savedVibes')) || [];
    
    // Update intensity display
    intensitySlider.addEventListener('input', () => {
        intensityValue.textContent = intensitySlider.value;
    });
    
    // Generate new vibe
    generateBtn.addEventListener('click', async () => {
        const vibeType = vibeTypeSelect.value;
        const intensity = intensitySlider.value;
        
        try {
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ type: vibeType, intensity })
            });
            
            if (!response.ok) {
                throw new Error('Failed to generate vibe');
            }
            
            currentVibe = await response.json();
            updateVibeDisplay();
            startVisualization();
            
            // Enable buttons
            playBtn.disabled = false;
            pauseBtn.disabled = false;
            saveBtn.disabled = false;
            
        } catch (error) {
            console.error('Error generating vibe:', error);
            alert('Failed to generate vibe. Please try again.');
        }
    });
    
    // Update the vibe display with current vibe info
    function updateVibeDisplay() {
        if (!currentVibe) return;
        
        currentVibeType.textContent = currentVibe.type.charAt(0).toUpperCase() + currentVibe.type.slice(1);
        vibeBpm.textContent = currentVibe.bpm;
        vibeKey.textContent = currentVibe.key;
        vibeScale.textContent = currentVibe.scale;
        vibeInstruments.textContent = currentVibe.instruments.join(', ');
        vibeEffects.textContent = currentVibe.effects.join(', ');
        
        // Apply color theme to visualizer background
        const visualizer = document.querySelector('.vibe-visualizer');
        visualizer.style.background = `linear-gradient(45deg, ${currentVibe.colors.map(c => c).join(', ')})`;
    }
    
    // Start the visualization
    function startVisualization() {
        if (animationId) {
            cancelAnimationFrame(animationId);
        }
        
        isPlaying = true;
        let particles = [];
        const particleCount = 50;
        
        // Create particles
        for (let i = 0; i < particleCount; i++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                size: Math.random() * 5 + 1,
                speedX: Math.random() * 3 - 1.5,
                speedY: Math.random() * 3 - 1.5,
                color: currentVibe.colors[Math.floor(Math.random() * currentVibe.colors.length)]
            });
        }
        
        // Animation function
        function animate() {
            if (!isPlaying) return;
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw and update particles
            particles.forEach(particle => {
                ctx.fillStyle = particle.color;
                ctx.beginPath();
                ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
                ctx.fill();
                
                // Update position
                particle.x += particle.speedX;
                particle.y += particle.speedY;
                
                // Bounce off edges
                if (particle.x < 0 || particle.x > canvas.width) {
                    particle.speedX *= -1;
                }
                
                if (particle.y < 0 || particle.y > canvas.height) {
                    particle.speedY *= -1;
                }
            });
            
            // Draw pulsing circles based on BPM
            const time = Date.now();
            const bpmMs = 60000 / currentVibe.bpm;
            const phase = (time % bpmMs) / bpmMs;
            
            const radius = 50 + Math.sin(phase * Math.PI * 2) * 20 * currentVibe.intensity;
            
            ctx.beginPath();
            ctx.arc(canvas.width / 2, canvas.height / 2, radius, 0, Math.PI * 2);
            ctx.strokeStyle = currentVibe.colors[0];
            ctx.lineWidth = 3;
            ctx.stroke();
            
            ctx.beginPath();
            ctx.arc(canvas.width / 2, canvas.height / 2, radius * 1.5, 0, Math.PI * 2);
            ctx.strokeStyle = currentVibe.colors[1] || currentVibe.colors[0];
            ctx.lineWidth = 2;
            ctx.stroke();
            
            animationId = requestAnimationFrame(animate);
        }
        
        animate();
    }
    
    // Play button
    playBtn.addEventListener('click', () => {
        if (!currentVibe) return;
        
        isPlaying = true;
        startVisualization();
    });
    
    // Pause button
    pauseBtn.addEventListener('click', () => {
        isPlaying = false;
        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }
    });
    
    // Save current vibe
    saveBtn.addEventListener('click', () => {
        if (!currentVibe) return;
        
        // Check if vibe already exists
        if (!savedVibes.some(vibe => vibe.id === currentVibe.id)) {
            savedVibes.push(currentVibe);
            localStorage.setItem('savedVibes', JSON.stringify(savedVibes));
            updateSavedVibesList();
        }
    });
    
    // Update the saved vibes list
    function updateSavedVibesList() {
        savedVibesList.innerHTML = '';
        
        if (savedVibes.length === 0) {
            const emptyItem = document.createElement('li');
            emptyItem.textContent = 'No saved vibes yet';
            emptyItem.style.opacity = '0.5';
            savedVibesList.appendChild(emptyItem);
            return;
        }
        
        savedVibes.forEach(vibe => {
            const vibeItem = document.createElement('li');
            vibeItem.innerHTML = `
                <strong>${vibe.type.charAt(0).toUpperCase() + vibe.type.slice(1)}</strong>
                <div>${vibe.key} ${vibe.scale}, ${vibe.bpm} BPM</div>
            `;
            
            // Add color indicator
            const colorIndicator = document.createElement('div');
            colorIndicator.style.height = '5px';
            colorIndicator.style.marginTop = '5px';
            colorIndicator.style.background = `linear-gradient(to right, ${vibe.colors.join(', ')})`;
            colorIndicator.style.borderRadius = '2px';
            vibeItem.appendChild(colorIndicator);
            
            // Load this vibe when clicked
            vibeItem.addEventListener('click', () => {
                currentVibe = vibe;
                updateVibeDisplay();
                startVisualization();
                
                // Enable buttons
                playBtn.disabled = false;
                pauseBtn.disabled = false;
                saveBtn.disabled = false;
            });
            
            savedVibesList.appendChild(vibeItem);
        });
    }
    
    // Initial load of saved vibes
    updateSavedVibesList();
    
    // Check if there's a current vibe on the server
    async function checkCurrentVibe() {
        try {
            const response = await fetch('/api/current');
            if (response.ok) {
                currentVibe = await response.json();
                updateVibeDisplay();
                startVisualization();
                
                // Enable buttons
                playBtn.disabled = false;
                pauseBtn.disabled = false;
                saveBtn.disabled = false;
            }
        } catch (error) {
            console.error('Error fetching current vibe:', error);
        }
    }
    
    // Check for current vibe on load
    checkCurrentVibe();
});

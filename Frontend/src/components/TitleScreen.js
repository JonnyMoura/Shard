import { logoIcon } from '../assets.js';

export function createTitleScreen(onContinue) {
    const titleScreen = document.createElement('div');
    titleScreen.id = 'title-screen';
    titleScreen.className = 'title-screen';

    const starsCanvas = document.createElement('canvas');
    starsCanvas.className = 'title-stars-canvas';
    starsCanvas.width = window.innerWidth;
    starsCanvas.height = window.innerHeight;

    const logoContainer = document.createElement('div');
    logoContainer.className = 'title-logo-container';

    const logoImg = document.createElement('img');
    logoImg.src = logoIcon;
    logoImg.alt = 'Logo';
    logoImg.className = 'title-logo';
    logoContainer.appendChild(logoImg);

    const pressMessage = document.createElement('div');
    pressMessage.className = 'title-press-message';
    pressMessage.textContent = 'Press anywhere to continue';

    titleScreen.appendChild(starsCanvas);
    titleScreen.appendChild(logoContainer);
    titleScreen.appendChild(pressMessage);
    document.body.appendChild(titleScreen);

    const context = starsCanvas.getContext('2d');
    const stars = [];
    const starsCount = 300;

    for (let i = 0; i < starsCount; i++) {
        stars.push({
            x: Math.random() * starsCanvas.width,
            y: Math.random() * starsCanvas.height,
            radius: Math.random() * 2 + 0.5,
            opacity: Math.random() * 0.7 + 0.3,
            speed: Math.random() * 0.3 + 0.1,
            twinkleSpeed: Math.random() * 0.02 + 0.01,
            twinklePhase: Math.random() * Math.PI * 2,
            color: Math.random() < 0.3
                ? 'rgb(150, 180, 255)'
                : Math.random() < 0.7
                    ? 'rgb(200, 220, 255)'
                    : 'rgb(255, 240, 200)'
        });
    }

    function animateStars() {
        const gradient = context.createRadialGradient(
            starsCanvas.width / 2,
            starsCanvas.height / 2,
            0,
            starsCanvas.width / 2,
            starsCanvas.height / 2,
            starsCanvas.width / 2
        );
        gradient.addColorStop(0, '#0a0a20');
        gradient.addColorStop(0.4, '#060a18');
        gradient.addColorStop(0.8, '#040815');
        gradient.addColorStop(1, '#020205');

        context.fillStyle = gradient;
        context.fillRect(0, 0, starsCanvas.width, starsCanvas.height);

        stars.forEach((star) => {
            star.twinklePhase += star.twinkleSpeed;
            const twinkle = Math.sin(star.twinklePhase) * 0.3 + 0.7;

            context.beginPath();
            context.arc(star.x, star.y, star.radius, 0, Math.PI * 2);
            context.fillStyle = star.color;
            context.globalAlpha = star.opacity * twinkle;
            context.fill();
        });

        context.globalAlpha = 1;
        if (titleScreen.parentElement) {
            requestAnimationFrame(animateStars);
        }
    }

    animateStars();

    const resizeHandler = () => {
        starsCanvas.width = window.innerWidth;
        starsCanvas.height = window.innerHeight;

        stars.forEach((star) => {
            if (star.x > starsCanvas.width) star.x = Math.random() * starsCanvas.width;
            if (star.y > starsCanvas.height) star.y = Math.random() * starsCanvas.height;
        });
    };
    window.addEventListener('resize', resizeHandler);

    let titleScreenRemoved = false;
    const removeTitleScreen = () => {
        if (titleScreenRemoved) return;
        titleScreenRemoved = true;

        titleScreen.classList.add('fade-out');
        setTimeout(() => {
            onContinue();
        }, 300);
        setTimeout(() => {
            titleScreen.remove();
            window.removeEventListener('resize', resizeHandler);
        }, 1500);
    };

    titleScreen.addEventListener('click', removeTitleScreen);
    window.addEventListener('keydown', removeTitleScreen, { once: true });
}

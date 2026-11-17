import * as THREE from 'three';
import { ParticleSystem } from './components/ParticleSystem';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EvolvingUI } from './components/EvolvingUI';
import { SaveMode } from './components/SaveMode';
import { Library } from './components/Library';
import { PythonCommunication } from './components/PythonCommunication.js';
import { SurroundController } from './audio/SurroundController.js';
import { LoadingScreen } from './components/LoadingScreen.js';
import { installKeyboardControls, keysPressed } from './components/KeyboardControls.js';
import { createNavigationStructure } from './components/Navigation.js';
import { createTitleScreen } from './components/TitleScreen.js';
import { createModeTutorialSystem } from './components/ModeTutorialSystem.js';
import { playIcon, stopIcon } from './assets.js';

import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';

import './styles.css';
installKeyboardControls();

let scene, camera, renderer, particleSystem, controls;
let composer, bloomPass, grainPass;
let staticFrequencyData;
let surroundController;
let saveMode;
let loadingScreen;
let lastAnimationTime = 0;
let animationAccumulator = 0;
const PHYSICS_STEP = 1 / 60;
// Materials for selective bloom
const BLOOM_SCENE = 1;
const bloomLayer = new THREE.Layers();
bloomLayer.set(BLOOM_SCENE);



function init() {
    createNavigationStructure();
    
    // Create the scene
    scene = new THREE.Scene();

    // Enhanced darker gradient background for more contrast
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const context = canvas.getContext('2d');
 
    const gradient = context.createRadialGradient(256, 256, 0, 256, 256, 512);
    gradient.addColorStop(0, '#0a0a20');   // Lighter center (was #050515)
    gradient.addColorStop(0.4, '#060a18'); // Lighter mid (was #030815)
    gradient.addColorStop(0.8, '#040815'); // Lighter outer (was #020520)
    gradient.addColorStop(1, '#020205');   // Very dark edge instead of pure black (was #000000)
    
    context.fillStyle = gradient;
    context.fillRect(0, 0, 512, 512);
    
    const backgroundTexture = new THREE.CanvasTexture(canvas);
    scene.background = backgroundTexture;

   
    const gridSize = 600;
    const gridDivisions = 60;
    
    // Create custom grid geometry
    const gridGeometry = new THREE.PlaneGeometry(gridSize, gridSize, gridDivisions, gridDivisions);
    
    // Fixed shader material for dynamic grid visibility
    const gridMaterial = new THREE.ShaderMaterial({
        uniforms: {
            time: { value: 0 },
            lightPositions: { value: new Array(60).fill(0) }, 
            lightIntensities: { value: new Array(20).fill(0) }, 
            maxLights: { value: 0 }, 
            gridColor: { value: new THREE.Color().setHSL(0.6, 0.5, 0.3) }, // INCREASED lightness from 0.3 to 0.4
            gridOpacity: { value: 0.45 } // INCREASED from 0.4 to 0.5
        },
        vertexShader: `
            varying vec2 vUv;
            varying vec3 vWorldPosition;
            
            void main() {
                vUv = uv;
                vec4 worldPosition = modelMatrix * vec4(position, 1.0);
                vWorldPosition = worldPosition.xyz;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `,
        fragmentShader: `
            uniform float time;
            uniform vec3 lightPositions[20];
            uniform float lightIntensities[20];
            uniform int maxLights;
            uniform vec3 gridColor;
            uniform float gridOpacity;
            
            varying vec2 vUv;
            varying vec3 vWorldPosition;
            
            void main() {
                // Create grid pattern
                vec2 grid = abs(fract(vUv * 60.0) - 0.5) / fwidth(vUv * 60.0);
                float line = min(grid.x, grid.y);
                float gridPattern = 1.0 - min(line, 1.0);
                
                // Calculate illumination from lights (particles/solids)
                float totalIllumination = 0.0;
                
                for(int i = 0; i < 20; i++) {
                    if(i >= maxLights) break;
                    
                    vec3 lightPos = lightPositions[i];
                    float lightIntensity = lightIntensities[i];
                    
                    if(lightIntensity > 0.0) {
                        float distance = length(vWorldPosition.xz - lightPos.xz);
                        float falloff = 1.0 / (1.0 + distance * distance * 0.015);
                        totalIllumination += falloff * lightIntensity;
                    }
                }
                
                // Show grid where there's illumination (no pulsing)
                float finalOpacity = gridPattern * totalIllumination * gridOpacity;
                
                gl_FragColor = vec4(gridColor, finalOpacity);
            }
        `,
        transparent: true,
        depthWrite: false,
        side: THREE.DoubleSide
    });
    
    const dynamicGrid = new THREE.Mesh(gridGeometry, gridMaterial);
    dynamicGrid.rotation.x = -Math.PI / 2; 
    dynamicGrid.position.y = -0.1; 
    scene.add(dynamicGrid);
    
    // Store reference for updates
    scene.userData.dynamicGrid = dynamicGrid;

   
    const lineGroup = new THREE.Group();
    
    // Create horizontal floating lines that move
    for (let i = 0; i < 25; i++) {
        const lineGeometry = new THREE.BufferGeometry();
        const linePositions = new Float32Array([
            -100, 0, 0,
            100, 0, 0
        ]);
        lineGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
        
        const lineMaterial = new THREE.LineBasicMaterial({
            color: new THREE.Color().setHSL(0.65, 0.25, 0.25 + Math.random() * 0.20), 
            transparent: true,
            opacity: 0.15 + Math.random() * 0.15 
        });
        
        const line = new THREE.Line(lineGeometry, lineMaterial);
        line.position.set(
            (Math.random() - 0.5) * 400,
            Math.random() * 100 - 20,
            (Math.random() - 0.5) * 400
        );
        line.rotation.y = Math.random() * Math.PI * 2;
        
        // Enhanced animation properties
        line.userData = {
            speed: Math.random() * 0.03 + 0.01,
            direction: Math.random() > 0.5 ? 1 : -1,
            oscillationSpeed: Math.random() * 0.5 + 0.3,
            oscillationAmplitude: Math.random() * 2 + 1, // This is the complete line
            originalY: line.position.y,
            movementType: 'horizontal' 
        };
        
        lineGroup.add(line);
    }
    
    // Create vertical floating lines that move
    for (let i = 0; i < 20; i++) {
        const lineGeometry = new THREE.BufferGeometry();
        const linePositions = new Float32Array([
            0, -50, 0,
            0, 50, 0
        ]);
        lineGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
        
        const lineMaterial = new THREE.LineBasicMaterial({
            color: new THREE.Color().setHSL(0.6, 0.4, 0.20 + Math.random() * 0.15),
            transparent: true,
            opacity: 0.10 + Math.random() * 0.25
        });
        
        const line = new THREE.Line(lineGeometry, lineMaterial);
        line.position.set(
            (Math.random() - 0.5) * 300,
            Math.random() * 80,
            (Math.random() - 0.5) * 300
        );
        
       
        line.userData = {
            speed: Math.random() * 0.02 + 0.008,
            direction: Math.random() > 0.5 ? 1 : -1,
            oscillationSpeed: Math.random() * 0.4 + 0.15,
            oscillationAmplitude: Math.random() * 3 + 1.5,
            originalX: line.position.x,
            movementType: 'vertical' 
        };
        
        lineGroup.add(line);
    }
    
    
    for (let i = 0; i < 15; i++) {
        const lineGeometry = new THREE.BufferGeometry();
        const linePositions = new Float32Array([
            -30, -30, 0,
            30, 30, 0
        ]);
        lineGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
        
        const lineMaterial = new THREE.LineBasicMaterial({
            color: new THREE.Color().setHSL(0.65, 0.2, 0.12 + Math.random() * 0.12),
            transparent: true,
            opacity: 0.1 + Math.random() * 0.2
        });
        
        const line = new THREE.Line(lineGeometry, lineMaterial);
        line.position.set(
            (Math.random() - 0.5) * 350,
            Math.random() * 60 + 10,
            (Math.random() - 0.5) * 350
        );
        line.rotation.z = Math.random() * Math.PI;
        
        // Enhanced animation properties
        line.userData = {
            speed: Math.random() * 0.025 + 0.005,
            direction: Math.random() > 0.5 ? 1 : -1,
            rotationSpeed: (Math.random() - 0.5) * 0.01,
            oscillationSpeed: Math.random() * 0.3 + 0.1,
            oscillationAmplitude: Math.random() * 4 + 2,
            originalZ: line.position.z,
            movementType: 'diagonal' 
        };
        
        lineGroup.add(line);
    }
    
    scene.add(lineGroup);
    scene.userData.floatingLines = lineGroup;

   
    const starsGeometry = new THREE.BufferGeometry();
    const starsCount = 3000;
    const starsPositions = new Float32Array(starsCount * 3);
    const starsColors = new Float32Array(starsCount * 3);
    
    for (let i = 0; i < starsCount * 3; i += 3) {
        // Distribute stars in a large sphere around the scene
        const radius = 400 + Math.random() * 800; 
        const theta = Math.random() * Math.PI * 2; 
        const phi = Math.random() * Math.PI; 
        
        starsPositions[i] = radius * Math.sin(phi) * Math.cos(theta);     // x
        starsPositions[i + 1] = radius * Math.cos(phi) - 200;            // y 
        starsPositions[i + 2] = radius * Math.sin(phi) * Math.sin(theta); // z
        
        // INCREASED brightness for all stars
        const colorVariation = Math.random();
        if (colorVariation < 0.3) {
            // Blue-white stars - BRIGHTER
            starsColors[i] = 0.6 + Math.random() * 0.3;     // R (was 0.4 + 0.2)
            starsColors[i + 1] = 0.7 + Math.random() * 0.3; // G (was 0.5 + 0.2)
            starsColors[i + 2] = 0.9 + Math.random() * 0.1; // B (was 0.7 + 0.2)
        } else if (colorVariation < 0.7) {
            // White stars - MUCH BRIGHTER
            const white = 0.85 + Math.random() * 0.15; // (was 0.6 + 0.2)
            starsColors[i] = white;     // R
            starsColors[i + 1] = white; // G
            starsColors[i + 2] = white; // B
        } else {
            // Slightly yellow stars - BRIGHTER
            starsColors[i] = 0.9 + Math.random() * 0.1;      // R (was 0.7 + 0.2)
            starsColors[i + 1] = 0.8 + Math.random() * 0.15; // G (was 0.6 + 0.15)
            starsColors[i + 2] = 0.6 + Math.random() * 0.2;  // B (was 0.4 + 0.15)
        }
    }
    
    starsGeometry.setAttribute('position', new THREE.BufferAttribute(starsPositions, 3));
    starsGeometry.setAttribute('color', new THREE.BufferAttribute(starsColors, 3));
    
    const starsMaterial = new THREE.PointsMaterial({
        size: 2.3,          // INCREASED from 1.2 to 2.8 (much larger)
        transparent: true,
        opacity: 0.7,      // INCREASED from 0.7 to 0.95 (almost opaque)
        vertexColors: true,
        fog: false,
        blending: THREE.AdditiveBlending, // NEW: makes stars glow
        depthWrite: false   // NEW: prevents z-fighting
    });
    
    const stars = new THREE.Points(starsGeometry, starsMaterial);
    scene.add(stars);
    
    // Store reference for subtle animation
    scene.userData.stars = stars;

    // Enhanced fog for depth
    scene.fog = new THREE.FogExp2(0x0a0a20, 0.002); // LIGHTER fog color and LESS dense (was 0x000308, 0.003)

 
    camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 2000);
    camera.position.set(0, 15, 30);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    document.querySelectorAll('canvas').forEach(c => c.remove());
    document.body.appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);

    
    // Enhanced lighting for better clarity
    const ambientLight = new THREE.AmbientLight(0x5060a0, 1.2); // INCREASED from 0.8 to 1.2
    scene.add(ambientLight);

    // Primary key light - brighter and more focused
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.0); // INCREASED from 0.6 to 1.0
    keyLight.position.set(0, 30, 25); 
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.width = 2048; // INCREASED resolution from 1024
    keyLight.shadow.mapSize.height = 2048;
    keyLight.shadow.camera.near = 1;
    keyLight.shadow.camera.far = 100;
    keyLight.shadow.bias = -0.0001; 
    scene.add(keyLight);

    // Fill light - stronger for better definition
    const fillLight = new THREE.DirectionalLight(0x6699ff, 0.8); // INCREASED from 0.5 to 0.8
    fillLight.position.set(-20, 20, -20);
    scene.add(fillLight);

    // Secondary fill light - brighter
    const fillLight2 = new THREE.DirectionalLight(0x8899ff, 0.7); // INCREASED from 0.4 to 0.7
    fillLight2.position.set(30, 15, -10);
    scene.add(fillLight2);

    // Point light - more intense for central illumination
    const pointLight = new THREE.PointLight(0x88aaff, 0.6, 100); // INCREASED from 0.3 to 0.6, range from 80 to 100
    pointLight.position.set(0, 25, 0); // INCREASED height from 20 to 25
    scene.add(pointLight);

    // Rim light - stronger for edge definition
    const rimLight = new THREE.DirectionalLight(0xaa99ff, 0.4); // INCREASED from 0.2 to 0.4
    rimLight.position.set(0, -5, 20);
    scene.add(rimLight);

    // Hemisphere light - brighter for overall fill
    const hemisphereLight = new THREE.HemisphereLight(0x7799cc, 0x443355, 0.7); // INCREASED from 0.4 to 0.7
    scene.add(hemisphereLight);

    // Wireframe lights - more visible
    const wireframeLight1 = new THREE.DirectionalLight(0x6688bb, 0.2); // INCREASED from 0.2 to 0.4
    wireframeLight1.position.set(40, 10, 40);
    scene.add(wireframeLight1);

    const wireframeLight2 = new THREE.DirectionalLight(0x8866bb, 0.4); // INCREASED from 0.2 to 0.4
    wireframeLight2.position.set(-40, 10, -40);
    scene.add(wireframeLight2);

    // Up light - stronger for base illumination
    const upLight = new THREE.DirectionalLight(0x5577bb, 0.3); // INCREASED from 0.15 to 0.3
    upLight.position.set(0, -10, 0);
    scene.add(upLight);

    // NEW: Add additional top-down lights for better overall clarity
    const topLight1 = new THREE.DirectionalLight(0x99aaff, 0.5);
    topLight1.position.set(15, 40, 15);
    scene.add(topLight1);

    const topLight2 = new THREE.DirectionalLight(0x9999ff, 0.5);
    topLight2.position.set(-15, 40, -15);
    scene.add(topLight2);

    scene.userData.keyLight = keyLight;
    scene.userData.pointLight = pointLight;

  

    composer = new EffectComposer(renderer);

    // Base render pass 
    const renderPass = new RenderPass(scene, camera);
    composer.addPass(renderPass);

    // Bloom pass
    bloomPass = new UnrealBloomPass(
        new THREE.Vector2(window.innerWidth, window.innerHeight)
    );
    bloomPass.threshold = 0.1;
    bloomPass.strength = 1.2;
    bloomPass.radius = 0.4;
    composer.addPass(bloomPass);

    // Film grain pass l
    const FilmGrainShader = {
        uniforms: {
            'tDiffuse': { value: 0.3 },
            'time': { value: 0.0 },
            'nIntensity': { value: 0.035 },  // Grain intensity
            'grayscale': { value: 0.5 }       
        },
        vertexShader: `
            varying vec2 vUv;
            void main() {
                vUv = uv;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `,
        fragmentShader: `
            uniform sampler2D tDiffuse;
            uniform float time;
            uniform float nIntensity;
            uniform bool grayscale;
            varying vec2 vUv;

            float random(vec2 co) {
                return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
            }

            void main() {
                vec4 color = texture2D(tDiffuse, vUv);
                
                // Film grain only
                float grain = random(vUv + time) * nIntensity;
                color.rgb += grain;
                
                // Optional vignette
                vec2 center = vUv - 0.5;
                float vignette = 1.0 - dot(center, center) * 0.3;
                color.rgb *= vignette;
                
                gl_FragColor = color;
            }
        `
    };

    grainPass = new ShaderPass(FilmGrainShader);
    composer.addPass(grainPass);

    // Resize handler
    window.addEventListener('resize', () => {
        const width = window.innerWidth;
        const height = window.innerHeight;
        
        renderer.setSize(width, height);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        
        composer.setSize(width, height);
    });

    // Create ParticleSystem WITHOUT initialization
    particleSystem = new ParticleSystem(scene);
    
    // Show initial loading screen
    loadingScreen = new LoadingScreen();
    loadingScreen.show('Waiting for data');
    
    // CHANGED: Pass room dimensions that match the scene scale
    surroundController = new SurroundController();
    staticFrequencyData = new Uint8Array(
        surroundController.getFrequencyBinCount()
    );
    
    // CRITICAL: Set camera for audio listener IMMEDIATELY after creating SurroundController
    surroundController.setCamera(camera);
    console.log('🎥 Camera attached to audio listener');
    
    // Track if initial data has been received
    let initialDataReceived = false;
    
    const pythonComm = new PythonCommunication(
    particleSystem,
    // 1. onGenerationData callback - HIDES loading screen after audio loads
    (data) => {
        const generationNum = data.generation || 0;
        
        // Update loading message while audio loads
        loadingScreen.updateMessage(`Loading generation ${generationNum} sounds`);
        
        // Set particle system (needed for audio setup)
        surroundController.setParticleSystem(particleSystem);
        
        // Load audio for new generation
        surroundController.loadAll(data.solutions).then(() => {
            console.log(`✅ Generation ${generationNum} audio loaded`);
            
            // Add a small delay for visual smoothness
            setTimeout(() => {
                loadingScreen.hide();
                
                // NEW: Hide submission loading indicator if visible
                if (window.modeManager?.evolve?.hideSubmissionLoading) {
                    window.modeManager.evolve.hideSubmissionLoading();
                }

                // Create soundscape toggle button (only first time)
                if (!initialDataReceived) {
                    initialDataReceived = true;
                    createSoundscapeToggle(surroundController);
                }
            }, 300); // 300ms delay for smooth transition
            
        }).catch(error => {
            console.error('❌ Failed to load audio:', error);
            loadingScreen.hide();
        });
    },
    // 2. onConnectionReady callback
    () => {
        if (!initialDataReceived) {
            loadingScreen.updateMessage('Requesting generation data');
        }
    },
    // 3. onRegenerationStart callback - SHOWS loading screen ⚠️ THIS WAS MISSING!
    (data) => {
        const generationNum = data.generation || 0;
        console.log(`🔄 Regeneration starting for generation ${generationNum}`);
        
        // Show loading screen immediately when regeneration starts
        const message = initialDataReceived 
            ? `Loading generation ${generationNum}`
            : 'Loading initial generation';
        
        loadingScreen.show(message);
    }
);
    
    window.pythonComm = pythonComm;

    // Initialize other components
    const library = new Library(particleSystem, scene, camera, renderer, controls);
    const evolvingUI = new EvolvingUI(renderer, camera, controls, particleSystem, pythonComm, library);
    saveMode = new SaveMode(renderer, camera, controls, particleSystem, library);

    // modes  routing ---
    const sc = surroundController;
    const modeManager = {
        evolve: evolvingUI,
        save: saveMode,
        library,

        _locked: false,

        isActive(name) {
            if (name === 'evolve') return !!this.evolve?.isEvolvingMode;
            if (name === 'save') return !!this.save?.isSaveMode;
            if (name === 'library') return !!this.library?.inSpace;
            return false;
        },

        // NEW: Helper to toggle soundscape button visibility
        _updateSoundscapeButton() {
            const btn = document.getElementById('soundscape-toggle-btn');
            if (!btn) return;
            
            // Hide if ANY mode is active, show only on main scene
            const anyModeActive = this.isActive('evolve') || this.isActive('save') || this.isActive('library');
            
            if (anyModeActive) {
                btn.classList.add('hidden');
            } else {
                btn.classList.remove('hidden');
            }
        },

        // REMOVED: _syncButtons() - no longer needed

        // Normalize small leftovers when returning to main
        async _closeToMain() {
            try { 
                // ADDED: Set audio back to main mode with spatial audio
                if (sc) {
                    await sc.setMode('main');
                }
                
                sc?.clearAllGreenOutlines && sc.clearAllGreenOutlines(); 
            } catch {}
            await new Promise(r => setTimeout(r, 0));
            this._updateSoundscapeButton();
        },

        // Exit a specific mode (idempotent), then lightly normalize to main
        async exit(name, { skipNormalize = false } = {}) {
            try {
                // CRITICAL: Stop ALL audio before exiting ANY mode
                console.log(`🔇 Stopping all audio before exiting ${name}`);
                try {
                    await sc?.stopAllManualPlayback?.();
                } catch (e) {
                    console.error('Failed to stop audio on mode exit:', e);
                }
                
                if (name === 'evolve' && this.evolve?.isEvolvingMode) {
                    await Promise.resolve(this.evolve.exitEvolvingMode());
                }
                if (name === 'save') {
                    this.save?.forceReset?.({ restoreMaterials: true });
                    if (this.save?.isSaveMode) {
                        await Promise.resolve(this.save.exitSaveMode());
                    }
                }
                if (name === 'library' && this.library?.inSpace) {
                    await Promise.resolve(this.library.exitSpace());
                }
            } catch (err) {
                console.warn(`Failed to exit mode "${name}"`, err);
            }

            if (!skipNormalize) {
                await this._closeToMain();
            }
        },

        async enter(name) {
            try {
                // NEW: Set audio mode FIRST (before entering mode)
                if (sc) {
                    await sc.setMode(name || 'main');
                }
                
                if (name && sc?.hasUserRequestedSoundscape?.()) {
                    console.log(`🔇 Stopping soundscape before entering ${name}`);
                    await sc.stopUserSoundscape(80);
                }
                
                if (name === 'evolve' && !this.evolve?.isEvolvingMode) {
                    await Promise.resolve(this.evolve.enterEvolvingMode());
                }
                if (name === 'save' && !this.save?.isSaveMode) {
                    await Promise.resolve(this.save.enterSaveMode());
                }
                if (name === 'library' && !this.library?.inSpace) {
                    await Promise.resolve(this.library.enterSpace());
                }

                // NEW: Show tutorial message for first-time mode entry
                if (name && window.modeTutorialSystem) {
                    // Delay slightly to let mode UI settle
                    setTimeout(() => {
                        window.modeTutorialSystem.show(name);
                    }, 800);
                }
            } catch (err) {
                console.warn(`Failed to enter mode "${name}"`, err);
            }

            this._updateSoundscapeButton();
        },

        async goTo(name) {
            if (this._locked) return;
            this._locked = true;

            window.__modeTransitioning = true;

            // NEW: Hide tutorial message immediately when changing modes
            if (window.modeTutorialSystem) {
                window.modeTutorialSystem.hide();
            }

            const wasTargetActive = name ? this.isActive(name) : false;
            const activeBefore = ['evolve', 'save', 'library'].filter(mode => this.isActive(mode));

            try {
                // Ensure Save mode is clean even if it wasn't marked active
                if (name !== 'save') {
                    this.save?.forceReset?.({ restoreMaterials: true });
                }

                for (const mode of activeBefore) {
                    await this.exit(mode, { skipNormalize: true });
                }

                await this._closeToMain();

                if (!name || wasTargetActive) {
                    return;
                }

                await this.enter(name);
            } finally {
                requestAnimationFrame(() => {
                    window.__modeTransitioning = false;
                });
                this._locked = false;
            }
        }
    };
    window.modeManager = modeManager;

    
    const BTN_SELECTORS = {
        evolve: '#evolve-btn',
        save: '#save-btn',
        library: '#library-btn',
    };

    function resolveBtnElement(btn, selector) {
        // Accept direct HTMLElement
        if (btn && typeof btn === 'object') {
            if (btn instanceof HTMLElement) return btn;
            if (btn.element instanceof HTMLElement) return btn.element;
            if (btn.el instanceof HTMLElement) return btn.el;
            if (typeof btn.getElement === 'function') {
                try {
                    const el = btn.getElement();
                    if (el instanceof HTMLElement) return el;
                } catch {}
            }
        }
        // Fallback: query by selector
        if (selector && typeof document !== 'undefined') {
            const el = document.querySelector(selector);
            if (el instanceof HTMLElement) return el;
        }
        return null;
    }

    // Centralized mode toggle handler
    const handleModeToggle = (mode) => {
        console.log(`🎯 Mode toggle requested: ${mode}`);
        
        // Just trigger mode change
        modeManager.goTo(mode);
    };

    // Attach handlers directly to button elements (one handler per button)
    const attachModeHandler = (btnObj, selector, mode) => {
        const el = resolveBtnElement(btnObj, selector);
        if (!el) {
            console.warn(`Could not resolve button element for ${mode}`);
            return;
        }
        
        // CRITICAL: Remove old handlers first
        const oldHandler = el._modeHandler;
        if (oldHandler) {
            el.removeEventListener('click', oldHandler, true);
        }
        
        // Remove inline onclick
        el.onclick = null;
        
        // Create new handler
        const handler = (e) => {
            e.preventDefault();
            e.stopPropagation();
            hoverHandlers?.hide();
            console.log(`🎯 Button clicked: ${mode}`);
            handleModeToggle(mode);
        };
        
        // Store reference to handler for later removal
        el._modeHandler = handler;
        
        // Add handler
        el.addEventListener('click', handler, { capture: true });
        
        // NEW: Add hover listeners for message display
        const hoverHandlers = window.navigationContainers?.hoverMessage;
        if (hoverHandlers) {
            el.addEventListener('mouseenter', () => {
                const mm = window.modeManager;
                const anyModeActive = mm?.isActive('save') || mm?.isActive('evolve') || mm?.isActive('library');
                if (!anyModeActive) {
                    hoverHandlers.show(mode);
                }
            });
            
            el.addEventListener('mouseleave', () => {
                hoverHandlers.hide();
            });
        }
        
        // Update button object reference if needed
        if (btnObj && typeof btnObj === 'object' && !(btnObj instanceof HTMLElement)) {
            if (btnObj.button) btnObj.button = el;
            if (btnObj.element) btnObj.element = el;
            if (btnObj.el) btnObj.el = el;
        }
        
        console.log(`✅ Attached mode handler for ${mode}`);
    };

    // Attach to each mode button
    attachModeHandler(evolvingUI?.evolveBtn, BTN_SELECTORS.evolve, 'evolve');
    attachModeHandler(saveMode?.saveBtn, BTN_SELECTORS.save, 'save');
    attachModeHandler(library?.libraryBtn, BTN_SELECTORS.library, 'library');
    // Remove the main-scene click-to-play handler to avoid interrupting sequential mode
    // setupSolidClickHandler(pythonComm, surroundController);

    // Expose library for debugging
    window.library = library;

    // CHANGED: Re-enable click-to-play with soundscape pause/resume
    setupSolidClickHandler(pythonComm, surroundController);

    animate();
}

// Camera movement with WASD/Arrow keys
function updateCameraMovement(deltaTime) {
    // Only move in main scene (not in special modes)
    const isSaveMode = window.modeManager?.isActive('save');
    const isEvolveMode = window.modeManager?.isActive('evolve');
    const isLibraryMode = window.modeManager?.isActive('library');

    if (isSaveMode || isEvolveMode || isLibraryMode) return;
    if (keysPressed.size === 0) return;

    const moveSpeed = 30 * deltaTime;
    const direction = new THREE.Vector3();

    // Get camera's forward and right vectors (on XZ plane)
    const forward = new THREE.Vector3();
    camera.getWorldDirection(forward);
    forward.y = 0;
    forward.normalize();

    const right = new THREE.Vector3();
    right.crossVectors(forward, new THREE.Vector3(0, 1, 0)).normalize();

    // Calculate movement direction based on pressed keys
    if (keysPressed.has('w') || keysPressed.has('arrowup')) {
        direction.add(forward);
    }
    if (keysPressed.has('s') || keysPressed.has('arrowdown')) {
        direction.sub(forward);
    }
    if (keysPressed.has('a') || keysPressed.has('arrowleft')) {
        direction.sub(right);
    }
    if (keysPressed.has('d') || keysPressed.has('arrowright')) {
        direction.add(right);
    }

    // Vertical movement (space = up, shift = down)
    const up = new THREE.Vector3(0, 1, 0);
    if (keysPressed.has(' ')) {
        direction.add(up);
    }
    if (keysPressed.has('shift')) {
        direction.sub(up);
    }

    // Apply movement
    if (direction.length() > 0) {
        direction.normalize().multiplyScalar(moveSpeed);
        camera.position.add(direction);
        controls.target.add(direction);
    }
}

function animate() {
    requestAnimationFrame(animate);
    const now = performance.now();
    const deltaTime = lastAnimationTime
        ? Math.min((now - lastAnimationTime) / 1000, 0.1)
        : PHYSICS_STEP;
    lastAnimationTime = now;

    // Skip updates and rendering while switching modes (prevents one-frame flash)
    if (window.__modeTransitioning) {
        animationAccumulator = 0;
        return;
    }

    // Update camera movement from keyboard
    updateCameraMovement(deltaTime);
    surroundController?.getFrequencyData(staticFrequencyData);
    animationAccumulator += deltaTime;
    while (animationAccumulator >= PHYSICS_STEP) {
        particleSystem.update(staticFrequencyData);
        animationAccumulator -= PHYSICS_STEP;
    }

    
    if (saveMode) {
        saveMode.update3DIconPositions();
    }

    // Keep library info panel pinned near the selected solid
    if (window.library && window.library.inSpace) {
        window.library.update();
    }

    // Update dynamic grid
    const time = performance.now() * 0.001;
    if (scene.userData.dynamicGrid) {
        scene.userData.dynamicGrid.material.uniforms.time.value = time;
    }

    // Update film grain
    if (grainPass) {
        grainPass.uniforms.time.value = time;
    }

    // Minimal light animation
    if (scene.userData.keyLight) {
        const lightRadius = 60;
        scene.userData.keyLight.position.x = Math.cos(time * 0.03) * lightRadius;
        scene.userData.keyLight.position.z = Math.sin(time * 0.03) * lightRadius;
    }

    // Animate floating lines 
    if (scene.userData.floatingLines) {
        scene.userData.floatingLines.children.forEach(line => {
            const userData = line.userData;
            
            if (userData.movementType === 'horizontal') {
                line.position.z += userData.speed * userData.direction;
                line.position.y = userData.originalY + 
                    Math.sin(time * userData.oscillationSpeed + line.position.x * 0.01) * userData.oscillationAmplitude;
                
                if (line.position.z > 200) line.position.z = -200;
                if (line.position.z < -200) line.position.z = 200;
                
            } else if (userData.movementType === 'vertical') {
                line.position.y += userData.speed * userData.direction;
                line.position.x = userData.originalX + 
                    Math.sin(time * userData.oscillationSpeed + line.position.y * 0.02) * userData.oscillationAmplitude;
                
                if (line.position.y > 100) line.position.y = -50;
                if (line.position.y < -50) line.position.y = 100;
                
            } else if (userData.movementType === 'diagonal') {
                line.position.z += userData.speed * userData.direction;
                line.position.y = userData.originalZ + 
                    Math.sin(time * userData.oscillationSpeed + line.position.z * 0.015) * userData.oscillationAmplitude;
                line.rotation.z += userData.rotationSpeed;
                
                if (line.position.z > 250) line.position.z = -250;
                if (line.position.z < -250) line.position.z = 250;
            }
            
            const basOpacity = userData.movementType === 'horizontal' ? 0.25 : 
                               userData.movementType === 'vertical' ? 0.2 : 0.15;
            line.material.opacity = basOpacity + Math.sin(time * 1.5 + line.position.x + line.position.z) * 0.1;
        });
    }

   
    if (scene.userData.stars) {
        scene.userData.stars.rotation.y += 0.0001;
        scene.userData.stars.material.size = 1.2 + Math.sin(time * 0.5) * 0.3;
    }

    // Update camera controls
    controls.update();

    // Clear renderer
    renderer.clear();

 

    // Render with bloom effect
    composer.render();
}



let lastClickedSolidId = null;
// ADDED: Expose globally so PythonCommunication can reset it
window.lastClickedSolidId = null;

function setupSolidClickHandler(pythonComm, surroundController) {
    window.addEventListener('click', async (event) => {
        // ADDED: Ignore clicks if in any special mode
        const isSaveMode = window.modeManager?.isActive('save');
        const isEvolveMode = window.modeManager?.isActive('evolve');
        const isLibraryMode = window.modeManager?.isActive('library');
        
        if (isSaveMode || isEvolveMode || isLibraryMode) {
            return;
        }
        
        // ADDED: Ignore clicks on UI elements
        const target = event.target;
        if (target.closest('button') || 
            target.closest('.hamburger-menu') || 
            target.closest('.hamburger-options') ||
            target.closest('.name-input-overlay') ||
            target.closest('.library-space-panel')) {
            return;
        }
        
        const mouse = new THREE.Vector2();
        mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
        mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(mouse, camera);
        
        const solids = particleSystem.getSolids();
        if (solids.length > 0) {
            const intersects = raycaster.intersectObjects(solids, true);
            
            if (intersects.length > 0) {
                const clickedObject = intersects[0].object;
                let solid = clickedObject;
                while (solid.parent && !solids.includes(solid)) {
                    solid = solid.parent;
                }
                
                if (solid.userData && solid.userData.solution && surroundController) {
                    const solutionId = solid.userData.solution.id;
                    
                    // Check if clicking the same solid
                    const isClickingSameSolid = window.lastClickedSolidId === solutionId;
                    
                    // If clicking same solid, deselect and stop
                    if (isClickingSameSolid) {
                        console.log(`🔲 Deselecting solid ${solutionId}`);
                        
                        try {
                            await surroundController.stopOneShotById(solutionId, 200);
                        } catch (err) {
                            console.warn(`Failed to stop one-shot ${solutionId}:`, err);
                        }
                        
                        window.lastClickedSolidId = null;
                        
                        const wasSoundscapePlaying = surroundController._userRequestedPlay;
                        if (wasSoundscapePlaying) {
                            console.log(`🔄 Resuming soundscape after deselection`);
                            setTimeout(async () => {
                                await surroundController.playAll(5, {
                                    userInitiated: true,
                                    force: true
                                });
                            }, 300);
                        }
                        
                        return;
                    }
                    
                    // FIXED: Capture soundscape state BEFORE stopping anything
                    const wasSoundscapePlaying = surroundController._seqActive;
                    const userWantedSoundscape = surroundController._userRequestedPlay;
                    
                    console.log(`🎵 Clicked solid ${solutionId}`);
                    console.log(`   Previous solid: ${window.lastClickedSolidId}`);
                    console.log(`   Soundscape was active: ${wasSoundscapePlaying}`);
                    
                    // CRITICAL: Stop previous solid FIRST and WAIT for it to complete
                    if (window.lastClickedSolidId != null) {
                        console.log(`🛑 Stopping previous solid ${window.lastClickedSolidId} before playing ${solutionId}`);
                        try {
                            await surroundController.stopOneShotById(window.lastClickedSolidId, 150);
                        } catch (err) {
                            console.warn(`Failed to stop previous one-shot ${window.lastClickedSolidId}:`, err);
                        }
                        // ADDED: Small delay to ensure cleanup completes
                        await new Promise(resolve => setTimeout(resolve, 50));
                    }
                    
                    // Stop soundscape if it was playing
                    if (wasSoundscapePlaying) {
                        surroundController.stopSequential();

                        const currentId = surroundController._currentId;
                        if (currentId != null) {
                            const currentSource = surroundController.sources.get(currentId);
                            if (currentSource && surroundController._isSourcePlaying(currentSource)) {
                                console.log(`   Stopping sequential sound: ${currentId}`);
                                surroundController._stopSource(currentSource, true);
                            }
                            surroundController._unhighlightSource(currentId);
                        }

                        surroundController.clearAllGreenOutlines();
                        surroundController._currentId = null;
                        surroundController._emitPlaybackState?.(false);
                        console.log('🛑 Soundscape stopped completely');
                        
                        // ADDED: Small delay after stopping soundscape
                        await new Promise(resolve => setTimeout(resolve, 50));
                    }

                    if (wasSoundscapePlaying) {
                        surroundController._userRequestedPlay = true;
                    }

                    // NOW play the new solid (after everything is stopped)
                    console.log(`▶️ Starting playback of solid ${solutionId}`);
                    surroundController.playOneShot(solutionId, {
                        highlightMesh: solid,
                        onEnded: () => {
                            console.log(`✅ Solid ${solutionId} finished playing`);

                            if (wasSoundscapePlaying && surroundController._userRequestedPlay) {
                                setTimeout(async () => {
                                    const hasManualPlayback = surroundController.hasActiveOneShots?.() || false;
        
                                    if (!hasManualPlayback) {
                                        console.log(`🔄 RESUMING soundscape`);
                                        await surroundController.playAll(5, {
                                            userInitiated: true,
                                            force: true
                                        });
                                    }
                                }, 1000);
                            }
                            
                            if (window.lastClickedSolidId === solutionId) {
                                window.lastClickedSolidId = null;
                            }
                        },
                        allowOverlap: false
                    });

                    window.lastClickedSolidId = solutionId;
                    console.log(`✅ Now tracking solid ${solutionId}`);
                }
            }
        }
    });
}


window.testPythonConnection = () => {
    console.log('🧪 MANUAL CONNECTION TEST');
    if (window.pythonComm) {
        window.pythonComm.logConnectionStatus();
    } else {
        console.error('❌ Python communication not available');
    }
};


window.sendTestMessage = () => {
    console.log('🧪 SENDING TEST MESSAGE TO PYTHON');
    if (window.pythonComm && window.pythonComm.isConnected()) {
        window.pythonComm.websocket.send(JSON.stringify({
            action: 'test',
            message: 'Hello from JavaScript!',
            timestamp: Date.now()
        }));
        console.log('✅ Test message sent');
    } else {
        console.error('❌ Not connected to Python');
    }
};

// Helper button because of chrome not auto play policy
function createSoundscapeToggle(surroundController) {
    // Remove any existing toggle
    const existing = document.getElementById('soundscape-toggle-btn');
    if (existing) existing.remove();

    const btn = document.createElement('button');
    btn.id = 'soundscape-toggle-btn';
    btn.className = 'soundscape-toggle-btn';
    btn.title = 'Play Soundscape';
    
    const img = document.createElement('img');
    img.src = playIcon;
    img.alt = 'Play Soundscape';
    btn.appendChild(img);

    let isPlaying = surroundController?.hasUserRequestedSoundscape?.() ?? false;

    const applyState = (playing) => {
        isPlaying = !!playing;
        if (isPlaying) {
            img.src = stopIcon;
            img.alt = 'Stop Soundscape';
            btn.title = 'Stop Soundscape';
        } else {
            img.src = playIcon;
            img.alt = 'Play Soundscape';
            btn.title = 'Play Soundscape';
        }
    };

    applyState(isPlaying);

    btn.addEventListener('click', async (e) => {
        e.preventDefault();
        try {
            if (!isPlaying) {
                await surroundController.playAll(5, { userInitiated: true });
                applyState(surroundController.hasUserRequestedSoundscape?.());
            } else {
                await surroundController.stopUserSoundscape(500);
                applyState(false);
            }
        } catch (err) {
            console.warn('Failed to toggle soundscape:', err);
            applyState(false);
        }
    });

    const stateHandler = (event) => {
        applyState(!!event.detail?.playing);
    };
    window.addEventListener('surround-playback-state', stateHandler);

    document.body.appendChild(btn);
    
    // NEW: Add hover events for showing message
    btn.addEventListener('mouseenter', () => {
        if (window.navigationContainers?.hoverMessage) {
            window.navigationContainers.hoverMessage.show('soundscape');
        }
    });
    
    btn.addEventListener('mouseleave', () => {
        if (window.navigationContainers?.hoverMessage) {
            window.navigationContainers.hoverMessage.hide();
        }
    });
    
    return btn;
}

// Initialize the application
createTitleScreen(init);
window.modeTutorialSystem = createModeTutorialSystem();
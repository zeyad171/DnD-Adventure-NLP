// ============================================
// D&D ADVENTURE - SVG Map Visualization
// ============================================

class MazeRenderer {
    constructor(svgElement, gameConfig) {
        this.svg = svgElement;
        this.config = gameConfig;
        this.roomSize = 60;
        this.roomGap = 40;
        this.offsetX = 100;
        this.offsetY = 50;
        
        // Room visual positions (adjusted for layout)
        this.roomPositions = {
            attic:   { x: 1, y: -1 },
            study:   { x: 0, y: 0 },
            library: { x: 1, y: 0 },
            gallery: { x: 2, y: 0 },
            cellar:  { x: 0, y: 1 },
            hall:    { x: 1, y: 1 },
            chapel:  { x: 2, y: 1 },
            crypt:   { x: 0, y: 2 },
            dungeon: { x: 1, y: 2 },
            vault:   { x: 2, y: 2 }
        };

        // Room icons - D&D themed
        this.roomIcons = {
            hall: '🍺',      // Tavern
            library: '🧙',    // Wizard's Study
            study: '💰',      // Treasure Chamber
            cellar: '⚒️',     // Dwarven Forge
            gallery: '⚔️',    // Hall of Heroes
            chapel: '☀️',     // Temple of Pelor
            attic: '🐉',      // Dragon's Hoard
            crypt: '💀',      // Tomb of the Lich
            dungeon: '👺',    // Goblin Prison
            vault: '🌀'       // Portal Chamber
        };

        this.connections = [
            ['hall', 'library'],
            ['hall', 'chapel'],
            ['hall', 'dungeon'],
            ['hall', 'cellar'],
            ['library', 'attic'],
            ['library', 'gallery'],
            ['library', 'study'],
            ['study', 'cellar'],
            ['gallery', 'chapel'],
            ['cellar', 'crypt'],
            ['crypt', 'dungeon'],
            ['dungeon', 'vault'],
            ['chapel', 'vault']
        ];

        this.init();
    }

    init() {
        // Add defs for gradients and filters
        this.addDefs();
        // Draw connections first (underneath rooms)
        this.drawConnections();
        // Draw rooms
        this.drawRooms();
    }

    addDefs() {
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        
        // Glow filter for current room
        defs.innerHTML = `
            <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
                <feMerge>
                    <feMergeNode in="coloredBlur"/>
                    <feMergeNode in="SourceGraphic"/>
                </feMerge>
            </filter>
            
            <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
                <feDropShadow dx="2" dy="2" stdDeviation="3" flood-color="#000" flood-opacity="0.5"/>
            </filter>
            
            <linearGradient id="roomGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#2a2a4a;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#1a1a2e;stop-opacity:1" />
            </linearGradient>
            
            <linearGradient id="currentGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#4a3000;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#2a1a00;stop-opacity:1" />
            </linearGradient>
            
            <linearGradient id="visitedGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#3a2060;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#2a1040;stop-opacity:1" />
            </linearGradient>
            
            <linearGradient id="lockedGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#4a1010;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#2a0505;stop-opacity:1" />
            </linearGradient>
        `;
        
        this.svg.appendChild(defs);
    }

    getPixelPosition(roomId) {
        const pos = this.roomPositions[roomId];
        return {
            x: this.offsetX + pos.x * (this.roomSize + this.roomGap),
            y: this.offsetY + pos.y * (this.roomSize + this.roomGap)
        };
    }

    drawConnections() {
        const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        group.setAttribute('id', 'connections');
        
        this.connections.forEach(([from, to]) => {
            const fromPos = this.getPixelPosition(from);
            const toPos = this.getPixelPosition(to);
            
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', fromPos.x + this.roomSize / 2);
            line.setAttribute('y1', fromPos.y + this.roomSize / 2);
            line.setAttribute('x2', toPos.x + this.roomSize / 2);
            line.setAttribute('y2', toPos.y + this.roomSize / 2);
            line.setAttribute('stroke', '#2a2a4a');
            line.setAttribute('stroke-width', '4');
            line.setAttribute('data-from', from);
            line.setAttribute('data-to', to);
            line.classList.add('connection-line');
            
            group.appendChild(line);
        });
        
        this.svg.appendChild(group);
    }

    drawRooms() {
        const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        group.setAttribute('id', 'rooms');
        
        Object.keys(this.roomPositions).forEach(roomId => {
            const pos = this.getPixelPosition(roomId);
            const roomGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            roomGroup.setAttribute('id', `room-${roomId}`);
            roomGroup.classList.add('room-node', 'unknown');
            roomGroup.setAttribute('data-room', roomId);
            
            // Room rectangle
            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            rect.setAttribute('x', pos.x);
            rect.setAttribute('y', pos.y);
            rect.setAttribute('width', this.roomSize);
            rect.setAttribute('height', this.roomSize);
            rect.setAttribute('rx', '8');
            rect.setAttribute('ry', '8');
            rect.setAttribute('fill', 'url(#roomGradient)');
            rect.setAttribute('stroke', '#3a3a5a');
            rect.setAttribute('stroke-width', '2');
            rect.classList.add('room-rect');
            
            // Room icon
            const icon = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            icon.setAttribute('x', pos.x + this.roomSize / 2);
            icon.setAttribute('y', pos.y + this.roomSize / 2 - 5);
            icon.setAttribute('text-anchor', 'middle');
            icon.setAttribute('dominant-baseline', 'middle');
            icon.setAttribute('font-size', '24');
            icon.textContent = this.roomIcons[roomId] || '❓';
            icon.classList.add('room-icon');
            
            // Room name
            const name = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            name.setAttribute('x', pos.x + this.roomSize / 2);
            name.setAttribute('y', pos.y + this.roomSize - 8);
            name.setAttribute('text-anchor', 'middle');
            name.setAttribute('font-size', '8');
            name.setAttribute('fill', '#888');
            name.setAttribute('font-family', 'Cinzel, serif');
            name.textContent = roomId.charAt(0).toUpperCase() + roomId.slice(1);
            name.classList.add('room-name');
            
            roomGroup.appendChild(rect);
            roomGroup.appendChild(icon);
            roomGroup.appendChild(name);
            
            // Click handler
            roomGroup.addEventListener('click', () => {
                if (window.gameState && typeof window.handleMapRoomClick === 'function') {
                    window.handleMapRoomClick(roomId);
                }
            });
            
            group.appendChild(roomGroup);
        });
        
        this.svg.appendChild(group);
        
        // Add exit indicator below vault
        this.addExitIndicator();
    }

    addExitIndicator() {
        const vaultPos = this.getPixelPosition('vault');
        const exitGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        exitGroup.setAttribute('id', 'exit-indicator');
        exitGroup.classList.add('hidden');
        
        // Line from vault to exit
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', vaultPos.x + this.roomSize / 2);
        line.setAttribute('y1', vaultPos.y + this.roomSize);
        line.setAttribute('x2', vaultPos.x + this.roomSize / 2);
        line.setAttribute('y2', vaultPos.y + this.roomSize + 40);
        line.setAttribute('stroke', '#ffd700');
        line.setAttribute('stroke-width', '3');
        line.setAttribute('stroke-dasharray', '5,5');
        
        // Exit text
        const exitText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        exitText.setAttribute('x', vaultPos.x + this.roomSize / 2);
        exitText.setAttribute('y', vaultPos.y + this.roomSize + 55);
        exitText.setAttribute('text-anchor', 'middle');
        exitText.setAttribute('font-size', '14');
        exitText.setAttribute('fill', '#ffd700');
        exitText.setAttribute('font-family', 'Cinzel, serif');
        exitText.textContent = '🚪 EXIT';
        
        exitGroup.appendChild(line);
        exitGroup.appendChild(exitText);
        this.svg.appendChild(exitGroup);
    }

    update(currentRoom, visitedRooms, unlockedDoors) {
        // Update all rooms
        Object.keys(this.roomPositions).forEach(roomId => {
            const roomGroup = document.getElementById(`room-${roomId}`);
            const rect = roomGroup.querySelector('.room-rect');
            
            // Remove all state classes
            roomGroup.classList.remove('current', 'visited', 'unknown', 'locked');
            
            if (roomId === currentRoom) {
                roomGroup.classList.add('current');
                rect.setAttribute('fill', 'url(#currentGradient)');
                rect.setAttribute('stroke', '#ffd700');
                rect.setAttribute('stroke-width', '3');
                rect.setAttribute('filter', 'url(#glow)');
            } else if (visitedRooms.includes(roomId)) {
                roomGroup.classList.add('visited');
                rect.setAttribute('fill', 'url(#visitedGradient)');
                rect.setAttribute('stroke', '#6b00a8');
                rect.setAttribute('stroke-width', '2');
                rect.removeAttribute('filter');
            } else {
                roomGroup.classList.add('unknown');
                rect.setAttribute('fill', 'url(#roomGradient)');
                rect.setAttribute('stroke', '#3a3a5a');
                rect.setAttribute('stroke-width', '2');
                rect.removeAttribute('filter');
            }
        });
        
        // Update connections
        document.querySelectorAll('.connection-line').forEach(line => {
            const from = line.getAttribute('data-from');
            const to = line.getAttribute('data-to');
            
            const fromVisited = visitedRooms.includes(from) || from === currentRoom;
            const toVisited = visitedRooms.includes(to) || to === currentRoom;
            
            if (fromVisited && toVisited) {
                line.setAttribute('stroke', '#6b00a8');
                line.setAttribute('stroke-width', '4');
            } else if (fromVisited || toVisited) {
                line.setAttribute('stroke', '#4a4a6a');
                line.setAttribute('stroke-width', '3');
            } else {
                line.setAttribute('stroke', '#2a2a4a');
                line.setAttribute('stroke-width', '2');
            }
        });

        // Show exit when vault is reached
        const exitIndicator = document.getElementById('exit-indicator');
        if (visitedRooms.includes('vault') || currentRoom === 'vault') {
            exitIndicator.classList.remove('hidden');
        }
    }

    highlightPath(fromRoom, toRoom) {
        // Animate a path between two rooms
        const fromPos = this.getPixelPosition(fromRoom);
        const toPos = this.getPixelPosition(toRoom);
        
        // Create animated dot
        const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        dot.setAttribute('r', '5');
        dot.setAttribute('fill', '#ffd700');
        dot.setAttribute('filter', 'url(#glow)');
        
        const animate = document.createElementNS('http://www.w3.org/2000/svg', 'animateMotion');
        animate.setAttribute('dur', '0.5s');
        animate.setAttribute('fill', 'freeze');
        animate.setAttribute('path', `M${fromPos.x + 30},${fromPos.y + 30} L${toPos.x + 30},${toPos.y + 30}`);
        
        dot.appendChild(animate);
        this.svg.appendChild(dot);
        
        // Remove after animation
        setTimeout(() => dot.remove(), 600);
    }

    pulseRoom(roomId) {
        const roomGroup = document.getElementById(`room-${roomId}`);
        if (roomGroup) {
            roomGroup.classList.add('pulse');
            setTimeout(() => roomGroup.classList.remove('pulse'), 1000);
        }
    }
}

// Export for use
if (typeof window !== 'undefined') {
    window.MazeRenderer = MazeRenderer;
}


// ============================================
// D&D ADVENTURE - GAME CONFIGURATION
// ============================================

const GAME_CONFIG = {
    // ============================================
    // ROOMS DEFINITION - D&D THEMED
    // ============================================
    rooms: {
        hall: {
            id: 'hall',
            name: 'Tavern Entrance',
            description: 'You stand in the entrance of the Dragon\'s Rest Tavern. The smell of ale and roasted meat fills the air. A quest board hangs on the wall, and passages lead to various parts of this mysterious establishment.',
            shortDesc: 'Where adventures begin',
            gridPosition: { row: 1, col: 1 },
            connections: {
                north: 'library',
                east: { room: 'chapel', requires: 'torch', locked: true },
                south: 'dungeon',
                west: 'cellar'
            },
            items: ['torch'],
            actions: ['look', 'take_torch'],
            isStart: true
        },
        library: {
            id: 'library',
            name: 'Wizard\'s Study',
            description: 'Arcane tomes and spell scrolls line the shelves of this magical study. A crystal ball glows softly on a reading desk. Runes etched into the floor pulse with magical energy.',
            shortDesc: 'Arcane knowledge awaits',
            gridPosition: { row: 0, col: 1 },
            connections: {
                south: 'hall',
                north: 'attic',
                east: { room: 'gallery', requires: 'candle', locked: true },
                west: { room: 'study', requires: 'rusty_key', locked: true }
            },
            items: ['candle', 'ancient_scroll'],
            actions: ['look', 'read_books', 'light_candles', 'take_candle', 'take_scroll'],
            puzzle: 'candle_sequence'
        },
        study: {
            id: 'study',
            name: 'Treasure Chamber',
            description: 'A hidden chamber filled with gold coins and precious gems. A portrait of a legendary dragon slayer watches over a locked chest containing untold riches.',
            shortDesc: 'Glittering treasures',
            gridPosition: { row: 0, col: 0 },
            connections: {
                east: 'library',
                south: 'cellar'
            },
            items: ['rusty_key'],
            actions: ['look', 'examine_portrait', 'open_safe', 'take_key'],
            puzzle: 'safe_combination'
        },
        cellar: {
            id: 'cellar',
            name: 'Dwarven Forge',
            description: 'The heat from ancient forges warms this stone chamber. Dwarven runes cover the walls, and weapon racks display legendary blades. A secret passage leads deeper underground.',
            shortDesc: 'Where weapons are born',
            gridPosition: { row: 1, col: 0 },
            connections: {
                north: 'study',
                east: 'hall',
                south: 'crypt'
            },
            items: [],
            actions: ['look']
        },
        gallery: {
            id: 'gallery',
            name: 'Hall of Heroes',
            description: 'Statues of legendary adventurers stand in eternal vigilance. Their stone eyes seem to judge your worth. A magical mirror shows visions of possible futures.',
            shortDesc: 'Legends remembered',
            gridPosition: { row: 0, col: 2 },
            connections: {
                west: 'library',
                south: { room: 'chapel', requires: 'torch', locked: true }
            },
            items: ['dagger'],
            actions: ['look', 'examine_mirror', 'examine_portraits', 'take_dagger'],
            puzzle: 'mirror_puzzle'
        },
        chapel: {
            id: 'chapel',
            name: 'Temple of Pelor',
            description: 'A sacred temple dedicated to the sun god. Golden light streams through stained glass windows. The altar radiates divine energy, offering blessings to worthy adventurers.',
            shortDesc: 'Divine sanctuary',
            gridPosition: { row: 1, col: 2 },
            connections: {
                north: 'gallery',
                west: { room: 'hall', requires: 'cross', locked: true },
                south: 'vault'
            },
            items: ['holy_water', 'cross'],
            actions: ['look', 'pray', 'examine_altar', 'take_holy_water', 'take_cross']
        },
        attic: {
            id: 'attic',
            name: 'Dragon\'s Hoard',
            description: 'Mountains of gold and magical artifacts fill this hidden chamber. Ancient dragon scales litter the floor. Somewhere here lies the key to your freedom.',
            shortDesc: 'Legendary treasures',
            gridPosition: { row: -1, col: 1 },
            connections: {
                south: 'library'
            },
            items: ['old_map'],
            actions: ['look', 'take_map']
        },
        crypt: {
            id: 'crypt',
            name: 'Tomb of the Lich',
            description: 'Ancient sarcophagi hold the remains of fallen heroes and dark lords. Necromantic energy crackles in the air. The inscriptions tell tales of a powerful lich who once ruled these lands.',
            shortDesc: 'Rest in peace... or not',
            gridPosition: { row: 2, col: 0 },
            connections: {
                north: 'cellar',
                east: 'dungeon'
            },
            items: ['skeleton_key'],
            actions: ['look', 'read_inscriptions', 'open_coffin', 'take_skeleton_key']
        },
        dungeon: {
            id: 'dungeon',
            name: 'Goblin Prison',
            description: 'Cages and chains fill this chamber where goblins once held their prisoners. A complex mechanism of levers controls the cell doors. Three magical locks guard the passage forward.',
            shortDesc: 'Freedom awaits',
            gridPosition: { row: 2, col: 1 },
            connections: {
                north: 'hall',
                west: 'crypt',
                east: { room: 'vault', requires: ['rusty_key', 'golden_key', 'skeleton_key'], locked: true }
            },
            items: ['golden_key'],
            actions: ['look', 'pull_lever', 'examine_mechanism', 'use_keys', 'take_golden_key'],
            puzzle: 'lever_mechanism'
        },
        vault: {
            id: 'vault',
            name: 'Portal Chamber',
            description: 'A mystical chamber where reality bends. Arcane circles glow with power, and a shimmering portal awaits activation. The Tome of Teleportation rests on a crystal pedestal.',
            shortDesc: 'Gateway to freedom',
            gridPosition: { row: 2, col: 2 },
            connections: {
                west: 'dungeon',
                north: 'chapel',
                exit: { room: 'escape', requires: 'ritual_complete', locked: true }
            },
            items: ['ancient_tome'],
            actions: ['look', 'take_tome', 'read_tome', 'perform_ritual', 'open_exit'],
            puzzle: 'final_ritual',
            isEnd: true
        }
    },

    // ============================================
    // ITEMS DEFINITION - D&D THEMED
    // ============================================
    items: {
        torch: {
            id: 'torch',
            name: 'Everburning Torch',
            description: 'A magical torch enchanted with continual flame. It never goes out.',
            icon: '🔥',
            usable: true,
            useAction: 'light_area'
        },
        candle: {
            id: 'candle',
            name: 'Ritual Candle',
            description: 'A mystical candle inscribed with arcane runes. Essential for spellcasting.',
            icon: '🕯️',
            usable: true,
            useAction: 'light_candle'
        },
        rusty_key: {
            id: 'rusty_key',
            name: 'Thieves\' Tools',
            description: 'A set of lockpicks favored by rogues. +2 to lockpicking checks.',
            icon: '🗝️',
            usable: true,
            useAction: 'unlock',
            unlocks: 'study'
        },
        golden_key: {
            id: 'golden_key',
            name: 'Dragon Scale Key',
            description: 'A key forged from dragon scales. Glows with ancient magic.',
            icon: '🔑',
            usable: true,
            useAction: 'unlock',
            unlocks: 'vault'
        },
        skeleton_key: {
            id: 'skeleton_key',
            name: 'Lich\'s Phylactery Key',
            description: 'A key carved from the bone of an ancient lich. Radiates necrotic energy.',
            icon: '💀',
            usable: true,
            useAction: 'unlock',
            unlocks: 'vault'
        },
        dagger: {
            id: 'dagger',
            name: 'Vorpal Dagger +1',
            description: 'A magical dagger that deals extra damage to undead. Critical on 19-20.',
            icon: '🗡️',
            usable: true,
            useAction: 'attack'
        },
        holy_water: {
            id: 'holy_water',
            name: 'Blessed Potion',
            description: 'Holy water blessed by a cleric of Pelor. Effective against undead.',
            icon: '💧',
            usable: true,
            useAction: 'purify'
        },
        cross: {
            id: 'cross',
            name: 'Holy Symbol',
            description: 'A sacred symbol of divine power. Grants advantage on saves vs evil.',
            icon: '✝️',
            usable: true,
            useAction: 'protect',
            unlocks: 'hall_chapel'
        },
        old_map: {
            id: 'old_map',
            name: 'Dungeon Map',
            description: 'A detailed map of the dungeon drawn by a previous adventurer.',
            icon: '🗺️',
            usable: true,
            useAction: 'reveal_map'
        },
        ancient_scroll: {
            id: 'ancient_scroll',
            name: 'Scroll of Teleportation',
            description: 'A spell scroll containing the teleportation circle ritual.',
            icon: '📜',
            usable: true,
            useAction: 'read_spell'
        },
        ancient_tome: {
            id: 'ancient_tome',
            name: 'Tome of Portal Magic',
            description: 'The legendary spellbook containing the portal activation ritual.',
            icon: '📕',
            usable: true,
            useAction: 'perform_ritual'
        }
    },

    // ============================================
    // PUZZLES DEFINITION - D&D THEMED
    // ============================================
    puzzles: {
        candle_sequence: {
            id: 'candle_sequence',
            name: 'Arcane Ritual Circle',
            description: 'Activate the ritual candles in the correct sequence to unlock the magical ward.',
            hint: 'The wizard\'s notes say: "Light from smallest flame to greatest power."',
            solution: [2, 0, 3, 1], // Candle indices in order
            solved: false,
            reward: 'Dispels the ward, revealing Thieves\' Tools'
        },
        safe_combination: {
            id: 'safe_combination',
            name: 'Dragon Slayer\'s Riddle',
            description: 'The portrait of the legendary hero holds the combination.',
            hint: 'Count the dragon heads slain in each section of the painting.',
            solution: '3-7-4-1',
            solved: false,
            reward: 'dragon_scale_key'
        },
        mirror_puzzle: {
            id: 'mirror_puzzle',
            name: 'Scrying Mirror Challenge',
            description: 'The magical mirror reveals secrets to those who prove their worth.',
            hint: 'Present your weapon at high noon position before the glass.',
            solution: 'dagger_midnight',
            solved: false,
            reward: 'Reveals secret passage to Temple'
        },
        lever_mechanism: {
            id: 'lever_mechanism',
            name: 'Goblin Lock Mechanism',
            description: 'Three enchanted levers control the prison cell doors.',
            hint: 'The lich\'s tomb inscriptions reveal: LEFT, RIGHT, MIDDLE.',
            solution: ['left', 'right', 'middle'],
            solved: false,
            reward: 'Opens passage to Portal Chamber'
        },
        final_ritual: {
            id: 'final_ritual',
            name: 'Portal Activation Ritual',
            description: 'Cast the teleportation ritual to activate the escape portal.',
            hint: 'Blessed potion on the circle, read the scroll, invoke the tome.',
            requirements: ['holy_water', 'ancient_scroll', 'ancient_tome'],
            solved: false,
            reward: 'Portal opens - FREEDOM!'
        }
    },

    // ============================================
    // ACTIONS DEFINITION
    // ============================================
    actions: {
        look: { id: 'look', name: 'Look Around', icon: '👁️', intent: 'inspect' },
        take_torch: { id: 'take_torch', name: 'Take Torch', icon: '🔥', intent: 'get_item', item: 'torch' },
        take_candle: { id: 'take_candle', name: 'Take Candle', icon: '🕯️', intent: 'get_item', item: 'candle' },
        take_key: { id: 'take_key', name: 'Take Key', icon: '🗝️', intent: 'get_item', item: 'rusty_key' },
        take_golden_key: { id: 'take_golden_key', name: 'Take Golden Key', icon: '🔑', intent: 'get_item', item: 'golden_key' },
        take_skeleton_key: { id: 'take_skeleton_key', name: 'Take Skeleton Key', icon: '💀', intent: 'get_item', item: 'skeleton_key' },
        take_dagger: { id: 'take_dagger', name: 'Take Dagger', icon: '🗡️', intent: 'get_item', item: 'dagger' },
        take_holy_water: { id: 'take_holy_water', name: 'Take Holy Water', icon: '💧', intent: 'get_item', item: 'holy_water' },
        take_cross: { id: 'take_cross', name: 'Take Cross', icon: '✝️', intent: 'get_item', item: 'cross' },
        take_map: { id: 'take_map', name: 'Take Map', icon: '🗺️', intent: 'get_item', item: 'old_map' },
        take_scroll: { id: 'take_scroll', name: 'Take Scroll', icon: '📜', intent: 'get_item', item: 'ancient_scroll' },
        take_tome: { id: 'take_tome', name: 'Take Tome', icon: '📕', intent: 'get_item', item: 'ancient_tome' },
        read_books: { id: 'read_books', name: 'Read Books', icon: '📚', intent: 'read' },
        read_inscriptions: { id: 'read_inscriptions', name: 'Read Inscriptions', icon: '📖', intent: 'read' },
        read_tome: { id: 'read_tome', name: 'Read Tome', icon: '📕', intent: 'read' },
        light_candles: { id: 'light_candles', name: 'Light Candles', icon: '🔥', intent: 'solve_puzzle' },
        examine_portrait: { id: 'examine_portrait', name: 'Examine Portrait', icon: '🖼️', intent: 'inspect' },
        examine_mirror: { id: 'examine_mirror', name: 'Examine Mirror', icon: '🪞', intent: 'solve_puzzle' },
        examine_altar: { id: 'examine_altar', name: 'Examine Altar', icon: '⛪', intent: 'inspect' },
        examine_portraits: { id: 'examine_portraits', name: 'Study Portraits', icon: '🎨', intent: 'inspect' },
        examine_mechanism: { id: 'examine_mechanism', name: 'Examine Mechanism', icon: '⚙️', intent: 'inspect' },
        examine_window: { id: 'examine_window', name: 'Look Outside', icon: '🪟', intent: 'inspect' },
        open_safe: { id: 'open_safe', name: 'Open Safe', icon: '🔐', intent: 'solve_puzzle' },
        open_coffin: { id: 'open_coffin', name: 'Open Coffin', icon: '⚰️', intent: 'inspect' },
        open_trapdoor: { id: 'open_trapdoor', name: 'Open Trapdoor', icon: '🚪', intent: 'navigate' },
        open_exit: { id: 'open_exit', name: 'Open Exit', icon: '🚪', intent: 'escape' },
        search_barrels: { id: 'search_barrels', name: 'Search Barrels', icon: '🛢️', intent: 'inspect' },
        search_chests: { id: 'search_chests', name: 'Search Chests', icon: '📦', intent: 'inspect' },
        pull_lever: { id: 'pull_lever', name: 'Pull Lever', icon: '🎚️', intent: 'solve_puzzle' },
        use_keys: { id: 'use_keys', name: 'Use Keys', icon: '🔑', intent: 'unlock' },
        pray: { id: 'pray', name: 'Pray', icon: '🙏', intent: 'interact' },
        perform_ritual: { id: 'perform_ritual', name: 'Perform Ritual', icon: '✨', intent: 'escape' },
        go_north: { id: 'go_north', name: 'Go North', icon: '⬆️', intent: 'navigate' },
        go_south: { id: 'go_south', name: 'Go South', icon: '⬇️', intent: 'navigate' },
        go_east: { id: 'go_east', name: 'Go East', icon: '➡️', intent: 'navigate' },
        go_west: { id: 'go_west', name: 'Go West', icon: '⬅️', intent: 'navigate' }
    },

    // ============================================
    // INTENTS FOR AI AGENT
    // ============================================
    intents: [
        'inspect',      // Look around, examine things
        'navigate',     // Move between rooms
        'get_item',     // Pick up items
        'use_item',     // Use an item from inventory
        'unlock',       // Unlock a door or container
        'read',         // Read books, scrolls, inscriptions
        'solve_puzzle', // Attempt to solve a puzzle
        'interact',     // Generic interaction
        'escape'        // Final escape action
    ],

    // ============================================
    // TRAINING CONFIGURATION
    // ============================================
    training: {
        maxSamples: 100,        // Max training samples to track
        learningThreshold: 20,  // Samples needed before "learned" status
        accuracyTarget: 0.8     // 80% accuracy = well-trained
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = GAME_CONFIG;
}


"""
Script to create a large PDF about the Animal Kingdom for testing
Uses PyMuPDF (fitz) which is already in requirements
"""
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("Warning: PyMuPDF not available. Install with: pip install pymupdf")

def create_animal_kingdom_pdf(output_path="animal_kingdom.pdf"):
    """Create a comprehensive PDF about the Animal Kingdom (10 pages)"""
    
    if not HAS_PYMUPDF:
        print("Error: PyMuPDF is required to create PDFs.")
        print("Please install it with: pip install pymupdf")
        return False
    
    doc = fitz.open()  # Create a new PDF
    
    # Page 1: Introduction to Animal Kingdom
    page1 = doc.new_page()
    text1 = """The Animal Kingdom: A Comprehensive Guide

Introduction

The animal kingdom, scientifically known as Animalia, is one of the most diverse and fascinating domains of life on Earth. With over 1.5 million described species and potentially millions more yet to be discovered, animals inhabit nearly every corner of our planet, from the deepest ocean trenches to the highest mountain peaks.

Animals are multicellular, eukaryotic organisms that are heterotrophic, meaning they obtain their nutrition by consuming other organisms. Unlike plants, animals cannot produce their own food through photosynthesis. They are characterized by their ability to move, respond to stimuli, and reproduce sexually in most cases.

Classification of Animals

The animal kingdom is divided into two main groups: vertebrates and invertebrates. Vertebrates have a backbone and include mammals, birds, reptiles, amphibians, and fish. Invertebrates lack a backbone and represent the vast majority of animal species, including insects, arachnids, mollusks, and many others.

Key Characteristics of Animals

1. Movement: Most animals can move at some stage of their life cycle, enabling them to find food, escape predators, and reproduce.

2. Sensory Systems: Animals have developed complex sensory systems to perceive their environment, including vision, hearing, smell, taste, and touch.

3. Nervous Systems: Animals possess nervous systems that allow them to process information and respond to stimuli in their environment.

4. Reproduction: Most animals reproduce sexually, though some can reproduce asexually. Many animals care for their young, ensuring their survival."""
    
    page1.insert_text((50, 50), text1, fontsize=10)
    
    # Page 2: Mammals
    page2 = doc.new_page()
    text2 = """Mammals: The Warm-Blooded Vertebrates

Overview

Mammals are a class of vertebrate animals characterized by several unique features. They are warm-blooded (endothermic), have hair or fur covering their bodies, and females produce milk to nourish their young. Mammals first appeared around 200 million years ago and have since diversified into over 6,000 species.

Key Characteristics

- Mammary Glands: All female mammals produce milk to feed their offspring
- Hair or Fur: Provides insulation and protection
- Warm-Blooded: Maintain constant body temperature
- Three Middle Ear Bones: Enhanced hearing capabilities
- Diaphragm: Efficient breathing mechanism
- Large Brain: Complex cognitive abilities

Major Groups of Mammals

1. Monotremes: Egg-laying mammals like the platypus and echidna
2. Marsupials: Pouched mammals like kangaroos, koalas, and opossums
3. Placentals: The largest group, including humans, dogs, whales, and elephants

Notable Mammal Species

- Blue Whale: The largest animal on Earth, reaching lengths of 100 feet
- African Elephant: The largest land mammal, with males weighing up to 6 tons
- Cheetah: The fastest land animal, capable of speeds up to 70 mph
- Bat: The only mammal capable of sustained flight
- Dolphin: Highly intelligent marine mammal with complex social structures

Habitats and Adaptations

Mammals have adapted to virtually every environment on Earth. Arctic mammals like polar bears have thick fur and fat layers for insulation. Desert mammals like camels can survive with minimal water. Aquatic mammals like whales have streamlined bodies and blubber for buoyancy and warmth."""
    
    page2.insert_text((50, 50), text2, fontsize=10)
    
    # Page 3: Birds
    page3 = doc.new_page()
    text3 = """Birds: Masters of the Sky

Introduction

Birds are feathered, warm-blooded vertebrates that belong to the class Aves. With over 10,000 known species, birds are found on every continent and in every type of habitat. They evolved from theropod dinosaurs approximately 150 million years ago and have since become one of the most successful groups of animals.

Unique Features

- Feathers: Lightweight structures that provide insulation and enable flight
- Beaks: Toothless jaws covered in keratin
- Lightweight Skeleton: Hollow bones reduce weight for flight
- High Metabolism: Maintain high body temperature (104-108°F)
- Four-Chambered Heart: Efficient oxygen circulation
- Egg-Laying: Hard-shelled eggs protect developing embryos

Flight Adaptations

Flight is the most distinctive feature of birds, though not all birds can fly. Flightless birds like ostriches, emus, and penguins have adapted to life on land or in water. Flying birds have:
- Powerful flight muscles attached to a keeled sternum
- Streamlined bodies to reduce air resistance
- Wings with specialized feathers for lift and control
- Excellent vision for navigation and hunting

Bird Behavior

- Migration: Many birds travel thousands of miles seasonally
- Nesting: Complex structures built for protection and warmth
- Communication: Songs, calls, and visual displays
- Social Behavior: Flocking, cooperative hunting, and communal roosting

Notable Bird Species

- Ostrich: Largest bird, reaching 9 feet tall and 300 pounds
- Hummingbird: Smallest bird, some species weigh less than 2 grams
- Peregrine Falcon: Fastest animal, diving at speeds over 200 mph
- Albatross: Longest wingspan, up to 11 feet
- Parrot: Highly intelligent, capable of tool use and complex communication"""
    
    page3.insert_text((50, 50), text3, fontsize=10)
    
    # Page 4: Reptiles
    page4 = doc.new_page()
    text4 = """Reptiles: The Scaled Vertebrates

Overview

Reptiles are cold-blooded (ectothermic) vertebrates that first appeared over 300 million years ago. They include snakes, lizards, turtles, crocodiles, and the tuatara. There are approximately 10,000 known species of reptiles, found on every continent except Antarctica.

Key Characteristics

- Scales: Protective keratin scales cover their bodies
- Cold-Blooded: Rely on external heat sources to regulate body temperature
- Egg-Laying: Most lay leathery-shelled eggs on land
- Lungs: Breathe air throughout their lives
- Three-Chambered Heart: Most have partially divided hearts (crocodiles have four chambers)

Major Groups

1. Squamata: Snakes and lizards (largest group with 9,000+ species)
2. Testudines: Turtles and tortoises (350+ species)
3. Crocodilia: Crocodiles, alligators, caimans, and gharials (25 species)
4. Rhynchocephalia: Tuatara (2 species, found only in New Zealand)

Adaptations

Reptiles have evolved numerous adaptations for survival:
- Camouflage: Many can change color or blend with their environment
- Venom: Some snakes produce potent toxins for hunting
- Regeneration: Some lizards can regrow lost tails
- Longevity: Some tortoises can live over 150 years
- Water Conservation: Efficient kidneys and scaly skin prevent water loss

Notable Reptile Species

- Saltwater Crocodile: Largest reptile, up to 23 feet long and 2,200 pounds
- Green Anaconda: Heaviest snake, can exceed 500 pounds
- Komodo Dragon: Largest lizard, up to 10 feet long
- Leatherback Sea Turtle: Largest turtle, up to 2,000 pounds
- Gila Monster: One of only two venomous lizards

Habitats

Reptiles inhabit diverse environments from deserts to rainforests, oceans to freshwater systems. Their ectothermic nature allows them to survive in extreme conditions where warm-blooded animals would struggle."""
    
    page4.insert_text((50, 50), text4, fontsize=10)
    
    # Page 5: Amphibians
    page5 = doc.new_page()
    text5 = """Amphibians: Life in Two Worlds

Introduction

Amphibians are cold-blooded vertebrates that typically live part of their lives in water and part on land. The name "amphibian" comes from the Greek meaning "double life." This group includes frogs, toads, salamanders, newts, and caecilians. There are approximately 8,000 known species.

Life Cycle

Most amphibians undergo metamorphosis:
1. Eggs: Laid in water, often in large clusters
2. Larval Stage: Aquatic, breathing through gills (tadpoles for frogs)
3. Metamorphosis: Transformation to adult form
4. Adult Stage: Terrestrial or semi-aquatic, breathing through lungs and skin

Key Characteristics

- Permeable Skin: Allows gas exchange and water absorption
- Dual Breathing: Lungs and/or gills, plus skin respiration
- Cold-Blooded: Ectothermic metabolism
- Three-Chambered Heart: Less efficient than four-chambered hearts
- Gelatinous Eggs: Lack hard shells, must stay moist

Major Groups

1. Anura: Frogs and toads (6,000+ species)
2. Caudata: Salamanders and newts (700+ species)
3. Gymnophiona: Caecilians (200+ species, limbless, burrowing)

Adaptations

- Poison Glands: Many produce toxins for defense
- Camouflage: Blend with environment to avoid predators
- Vocalization: Frogs use calls for mating and territory
- Regeneration: Some salamanders can regrow limbs
- Hibernation: Survive cold winters in dormant state

Notable Amphibian Species

- Goliath Frog: Largest frog, up to 3.3 feet long and 7 pounds
- Golden Poison Frog: Most toxic vertebrate, enough poison to kill 10 humans
- Axolotl: Unique salamander that retains larval features throughout life
- Chinese Giant Salamander: Largest amphibian, up to 5.9 feet long
- Glass Frog: Transparent skin reveals internal organs

Conservation Concerns

Amphibians are among the most threatened animal groups, with over 40% of species at risk. Threats include habitat loss, climate change, pollution, and the chytrid fungus disease."""
    
    page5.insert_text((50, 50), text5, fontsize=10)
    
    # Page 6: Fish
    page6 = doc.new_page()
    text6 = """Fish: The Aquatic Vertebrates

Overview

Fish are aquatic vertebrates that breathe through gills and have fins for movement. They are the most diverse group of vertebrates, with over 34,000 known species. Fish have been on Earth for over 500 million years and represent the ancestors of all terrestrial vertebrates.

Key Characteristics

- Gills: Extract oxygen from water
- Fins: Paired and unpaired fins for stability and propulsion
- Scales: Protective covering (in most species)
- Swim Bladder: Buoyancy control organ (in bony fish)
- Lateral Line: Sensory system detecting water movement
- Cold-Blooded: Ectothermic metabolism

Major Groups

1. Bony Fish (Osteichthyes): Largest group with 30,000+ species
   - Ray-finned fish: Most common (salmon, tuna, bass)
   - Lobe-finned fish: Includes lungfish and coelacanths

2. Cartilaginous Fish (Chondrichthyes): 1,200+ species
   - Sharks, rays, skates, and chimaeras
   - Skeleton made of cartilage, not bone

3. Jawless Fish (Agnatha): 120+ species
   - Lampreys and hagfish
   - Most primitive fish group

Adaptations

- Schooling: Many fish form large groups for protection
- Camouflage: Blend with surroundings to avoid predators
- Bioluminescence: Deep-sea fish produce their own light
- Electric Organs: Some fish generate electricity for navigation and hunting
- Venom: Many species have venomous spines or fangs

Notable Fish Species

- Whale Shark: Largest fish, up to 40 feet long and 20 tons
- Paedocypris: Smallest fish, less than 0.3 inches long
- Bluefin Tuna: Fastest fish, speeds up to 43 mph
- Coelacanth: "Living fossil," thought extinct until 1938
- Anglerfish: Deep-sea predator with bioluminescent lure

Habitats

Fish inhabit diverse aquatic environments:
- Freshwater: Rivers, lakes, streams, ponds
- Marine: Oceans, from shallow reefs to deep trenches
- Brackish: Estuaries where fresh and saltwater mix
- Extreme: Some survive in hot springs, frozen waters, or high-pressure depths"""
    
    page6.insert_text((50, 50), text6, fontsize=10)
    
    # Page 7: Invertebrates - Insects
    page7 = doc.new_page()
    text7 = """Insects: The Most Diverse Animal Group

Introduction

Insects are the most successful and diverse group of animals on Earth, with over 1 million described species and potentially millions more undiscovered. They belong to the phylum Arthropoda and class Insecta. Insects have been on Earth for over 400 million years and play crucial roles in ecosystems worldwide.

Key Characteristics

- Three Body Segments: Head, thorax, and abdomen
- Six Legs: Three pairs of jointed legs
- Exoskeleton: Hard outer covering made of chitin
- Compound Eyes: Multiple lenses providing wide field of vision
- Antennae: Sensory organs for touch, smell, and taste
- Wings: Most adult insects have one or two pairs (though not all can fly)

Life Cycle

Most insects undergo complete or incomplete metamorphosis:
- Complete: Egg → Larva → Pupa → Adult (butterflies, beetles, flies)
- Incomplete: Egg → Nymph → Adult (grasshoppers, cockroaches, dragonflies)

Major Orders

1. Coleoptera (Beetles): 400,000+ species - largest order
2. Lepidoptera (Butterflies & Moths): 180,000+ species
3. Hymenoptera (Bees, Wasps, Ants): 150,000+ species
4. Diptera (Flies): 150,000+ species
5. Hemiptera (True Bugs): 80,000+ species
6. Orthoptera (Grasshoppers, Crickets): 20,000+ species

Ecological Importance

- Pollination: Essential for reproduction of many plants
- Decomposition: Break down dead organic matter
- Food Source: Base of many food chains
- Pest Control: Some insects control pest populations
- Disease Vectors: Some transmit diseases (mosquitoes, ticks)

Notable Insect Species

- Goliath Beetle: One of the largest insects, up to 4.3 inches
- Atlas Moth: Largest moth, wingspan up to 12 inches
- Bullet Ant: Most painful insect sting
- Honeybee: Produces honey and essential for pollination
- Monarch Butterfly: Migrates thousands of miles annually

Adaptations

- Mimicry: Resemble other animals or objects for protection
- Social Behavior: Complex colonies (ants, bees, termites)
- Chemical Defense: Produce toxins or unpleasant odors
- Flight: Most efficient form of animal locomotion
- Metamorphosis: Allows exploitation of different resources at different life stages"""
    
    page7.insert_text((50, 50), text7, fontsize=10)
    
    # Page 8: Marine Invertebrates
    page8 = doc.new_page()
    text8 = """Marine Invertebrates: Life in the Oceans

Overview

Marine invertebrates represent the vast majority of ocean life. These animals lack backbones and include an incredible diversity of forms, from microscopic plankton to giant squids. They play fundamental roles in marine ecosystems and have existed for hundreds of millions of years.

Major Groups

1. Mollusks: 85,000+ species including:
   - Cephalopods: Octopuses, squids, cuttlefish, nautiluses
   - Bivalves: Clams, oysters, mussels, scallops
   - Gastropods: Snails, slugs, sea slugs
   - Chitons: Armored marine mollusks

2. Arthropods (Marine): Crustaceans including:
   - Crabs, lobsters, shrimp, krill
   - Barnacles, copepods, isopods

3. Cnidarians: 11,000+ species:
   - Jellyfish, corals, sea anemones, hydras
   - Stinging cells (cnidocytes) for defense and hunting

4. Echinoderms: 7,000+ species:
   - Starfish, sea urchins, sand dollars, sea cucumbers
   - Radial symmetry and water vascular system

5. Sponges: 8,500+ species:
   - Simplest multicellular animals
   - Filter feeders, important for water filtration

Notable Species

- Giant Squid: Largest invertebrate, up to 43 feet long
- Blue Whale's Diet: Krill, tiny crustaceans, up to 4 tons per day
- Portuguese Man o' War: Dangerous jellyfish with 30-foot tentacles
- Crown-of-Thorns Starfish: Can devastate coral reefs
- Giant Clam: Largest bivalve, up to 4 feet across and 500 pounds

Adaptations

- Camouflage: Many can change color and texture
- Regeneration: Starfish can regrow lost arms
- Symbiosis: Clownfish and sea anemones, cleaner shrimp
- Bioluminescence: Many deep-sea species produce light
- Hard Shells: Protection from predators

Ecological Roles

- Coral Reefs: Built by tiny coral polyps, support 25% of marine life
- Filter Feeding: Many invertebrates clean the water
- Food Web: Base of marine food chains
- Carbon Cycling: Important for ocean carbon balance
- Habitat Creation: Provide homes for other marine life"""
    
    page8.insert_text((50, 50), text8, fontsize=10)
    
    # Page 9: Animal Behavior and Communication
    page9 = doc.new_page()
    text9 = """Animal Behavior and Communication

Introduction

Animal behavior encompasses all the ways animals interact with their environment, other animals, and their own species. Understanding behavior helps us appreciate the complexity and intelligence of the animal kingdom. Communication is essential for survival, reproduction, and social organization.

Types of Communication

1. Visual Communication:
   - Body postures and movements
   - Color displays and patterns
   - Facial expressions
   - Examples: Peacock displays, bee dances, threat postures

2. Auditory Communication:
   - Vocalizations (songs, calls, clicks)
   - Sound production through body parts
   - Examples: Bird songs, whale songs, frog calls, cricket chirps

3. Chemical Communication:
   - Pheromones for mating and territory
   - Scent marking
   - Examples: Ant trails, cat marking, moth mating signals

4. Tactile Communication:
   - Touch and physical contact
   - Grooming behaviors
   - Examples: Primates grooming, elephant trunk touching

Social Behaviors

- Pack Hunting: Wolves, lions, orcas coordinate to hunt
- Migration: Seasonal movements over vast distances
- Hibernation: Winter dormancy to conserve energy
- Tool Use: Crows, otters, primates use objects as tools
- Play: Young animals learn through play behavior
- Altruism: Some animals help others at personal cost

Intelligence and Learning

- Problem Solving: Many animals can solve complex puzzles
- Memory: Elephants remember locations and individuals for decades
- Imitation: Some species learn by observing others
- Culture: Knowledge passed between generations (whales, primates)
- Self-Awareness: Some animals recognize themselves in mirrors

Mating Behaviors

- Courtship Displays: Elaborate rituals to attract mates
- Territorial Defense: Protecting breeding areas
- Parental Care: Varies from none to extensive
- Monogamy vs. Polygamy: Different reproductive strategies
- Sexual Selection: Traits that increase mating success

Notable Examples

- Honeybee Waggle Dance: Communicates direction and distance to food
- Dolphin Echolocation: Uses sound to navigate and hunt
- Octopus Problem-Solving: Can open jars and solve mazes
- Elephant Memory: Remembers locations and family members for 50+ years
- Chimpanzee Tool Use: Uses sticks to extract termites, stones to crack nuts"""
    
    page9.insert_text((50, 50), text9, fontsize=10)
    
    # Page 10: Conservation and Threats
    page10 = doc.new_page()
    text10 = """Animal Conservation and Threats

The Current Crisis

The Earth is experiencing a biodiversity crisis, with species disappearing at rates 100 to 1,000 times higher than natural extinction rates. Scientists estimate that we may be losing species faster than we can discover them. Conservation efforts are critical to preserve the incredible diversity of the animal kingdom.

Major Threats

1. Habitat Loss and Fragmentation:
   - Deforestation destroys millions of acres annually
   - Urbanization reduces available habitat
   - Agricultural expansion eliminates natural areas
   - Fragmentation isolates populations, reducing genetic diversity

2. Climate Change:
   - Rising temperatures affect migration patterns
   - Ocean acidification harms marine life
   - Extreme weather events destroy habitats
   - Shifting seasons disrupt breeding cycles

3. Pollution:
   - Plastic pollution kills millions of marine animals
   - Chemical pollutants accumulate in food chains
   - Oil spills devastate ecosystems
   - Light and noise pollution disrupt behaviors

4. Overexploitation:
   - Overfishing depletes fish populations
   - Poaching threatens many species
   - Illegal wildlife trade worth billions annually
   - Trophy hunting reduces populations

5. Invasive Species:
   - Non-native species outcompete natives
   - Predators eliminate local species
   - Diseases spread by introduced animals
   - Ecosystem disruption

Conservation Success Stories

- Bald Eagle: Recovered from near extinction, removed from endangered list
- Gray Wolf: Successfully reintroduced to Yellowstone National Park
- Giant Panda: Population increased through protected reserves
- California Condor: Brought back from 27 individuals through captive breeding
- Humpback Whale: Populations recovered after whaling bans

Conservation Strategies

- Protected Areas: National parks, wildlife reserves, marine sanctuaries
- Captive Breeding: Zoos and breeding programs for endangered species
- Habitat Restoration: Rebuilding damaged ecosystems
- Legislation: Laws protecting endangered species and habitats
- Community Involvement: Local people as conservation partners
- Research and Monitoring: Understanding populations and threats

What You Can Do

- Support Conservation Organizations: Donate to reputable wildlife groups
- Reduce, Reuse, Recycle: Minimize waste and plastic use
- Choose Sustainable Products: Support eco-friendly companies
- Educate Others: Spread awareness about conservation issues
- Reduce Carbon Footprint: Help combat climate change
- Support Protected Areas: Visit and support national parks and reserves

The Future

The fate of Earth's animals depends on our actions today. With dedicated conservation efforts, we can preserve the incredible diversity of the animal kingdom for future generations. Every species plays a role in the complex web of life, and losing any species diminishes our planet's richness and resilience."""
    
    page10.insert_text((50, 50), text10, fontsize=10)
    
    doc.save(output_path)
    doc.close()
    file_size = len(open(output_path, 'rb').read())
    print(f"✅ Created Animal Kingdom PDF: {output_path}")
    print(f"   File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
    print(f"   Pages: 10")
    return True

if __name__ == "__main__":
    import sys
    output_path = sys.argv[1] if len(sys.argv) > 1 else "../documents/animal_kingdom.pdf"
    success = create_animal_kingdom_pdf(output_path)
    if not success:
        sys.exit(1)


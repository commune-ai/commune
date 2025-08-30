 # start of file
import { useEffect, useState } from 'react';
import Head from 'next/head';
import { 
  Box, 
  Container, 
  Heading, 
  Text, 
  SimpleGrid, 
  Flex,
  Select,
  Input,
  InputGroup,
  InputLeftElement,
  Button,
  Stack
} from '@chakra-ui/react';
import { SearchIcon } from '@chakra-ui/icons';
import Header from '../components/Header';
import Footer from '../components/Footer';
import PropertyCard from '../components/PropertyCard';
import { useWeb3 } from '../hooks/useWeb3';

export default function Properties() {
  const { isConnected, connect } = useWeb3();
  const [properties, setProperties] = useState([]);
  const [filteredProperties, setFilteredProperties] = useState([]);
  const [filters, setFilters] = useState({
    minPrice: '',
    maxPrice: '',
    bedrooms: '',
    search: ''
  });
  
  useEffect(() => {
    // In a real app, we would fetch properties from the blockchain or API
    const sampleProperties = [
      {
        id: 1,
        address: '123 Main St, Anytown, USA',
        price: 350000,
        bedrooms: 3,
        bathrooms: 2,
        squareFeet: 2000,
        image: '/images/house1.jpg',
        tokenAddress: '0x...',
      },
      {
        id: 2,
        address: '456 Oak Ave, Somewhere, USA',
        price: 425000,
        bedrooms: 4,
        bathrooms: 2.5,
        squareFeet: 2400,
        image: '/images/house2.jpg',
        tokenAddress: '0x...',
      },
      {
        id: 3,
        address: '789 Pine Rd, Elsewhere, USA',
        price: 275000,
        bedrooms: 2,
        bathrooms: 1,
        squareFeet: 1500,
        image: '/images/house3.jpg',
        tokenAddress: '0x...',
      },
      {
        id: 4,
        address: '101 Cedar Ln, Nowhere, USA',
        price: 550000,
        bedrooms: 5,
        bathrooms: 3,
        squareFeet: 3200,
        image: '/images/house4.jpg',
        tokenAddress: '0x...',
      },
      {
        id: 5,
        address: '202 Maple Dr, Anywhere, USA',
        price: 320000,
        bedrooms: 3,
        bathrooms: 2,
        squareFeet: 1800,
        image: '/images/house5.jpg',
        tokenAddress: '0x...',
      },
      {
        id: 6,
        address: '303 Birch Blvd, Someplace, USA',
        price: 395000,
        bedrooms: 4,
        bathrooms: 2,
        squareFeet: 2200,
        image: '/images/house6.jpg',
        tokenAddress: '0x...',
      },
    ];
    
    setProperties(sampleProperties);
    setFilteredProperties(sampleProperties);
  }, []);
  
  useEffect(() => {
    // Apply filters when they change
    let results = [...properties];
    
    if (filters.minPrice) {
      results = results.filter(p => p.price >= parseInt(filters.minPrice));
    }
    
    if (filters.maxPrice) {
      results = results.filter(p => p.price <= parseInt(filters.maxPrice));
    }
    
    if (filters.bedrooms) {
      results = results.filter(p => p.bedrooms >= parseInt(filters.bedrooms));
    }
    
    if (filters.search) {
      const searchLower = filters.search.toLowerCase();
      results = results.filter(p => 
        p.address.toLowerCase().includes(searchLower)
      );
    }
    
    setFilteredProperties(results);
  }, [filters, properties]);
  
  const handleFilterChange = (e) => {
    const { name, value } = e.target;
    setFilters(prev => ({
      ...prev,
      [name]: value
    }));
  };
  
  const clearFilters = () => {
    setFilters({
      minPrice: '',
      maxPrice: '',
      bedrooms: '',
      search: ''
    });
  };
  
  return (
    <>
      <Head>
        <title>Properties | Home2Home</title>
        <meta name="description" content="Browse available properties on the Home2Home platform" />
      </Head>
      
      <Box minH="100vh" display="flex" flexDirection="column">
        <Header />
        
        <Container maxW="container.xl" py={8} flex="1">
          <Heading as="h1" size="xl" mb={8}>Available Properties</Heading>
          
          <Box mb={8} p={6} borderWidth="1px" borderRadius="lg" bg="white">
            <Stack direction={{ base: 'column', md: 'row' }} spacing={4} mb={4}>
              <InputGroup>
                <InputLeftElement pointerEvents="none">
                  <SearchIcon color="gray.300" />
                </InputLeftElement>
                <Input 
                  placeholder="Search by address" 
                  name="search"
                  value={filters.search}
                  onChange={handleFilterChange}
                />
              </InputGroup>
              
              <Select 
                placeholder="Min Bedrooms" 
                name="bedrooms"
                value={filters.bedrooms}
                onChange={handleFilterChange}
              >
                <option value="1">1+</option>
                <option value="2">2+</option>
                <option value="3">3+</option>
                <option value="4">4+</option>
                <option value="5">5+</option>
              </Select>
            </Stack>
            
            <Stack direction={{ base: 'column', md: 'row' }} spacing={4}>
              <Input 
                placeholder="Min Price" 
                type="number" 
                name="minPrice"
                value={filters.minPrice}
                onChange={handleFilterChange}
              />
              <Input 
                placeholder="Max Price" 
                type="number" 
                name="maxPrice"
                value={filters.maxPrice}
                onChange={handleFilterChange}
              />
              <Button onClick={clearFilters} variant="outline">
                Clear Filters
              </Button>
            </Stack>
          </Box>
          
          {filteredProperties.length > 0 ? (
            <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={10}>
              {filteredProperties.map((property) => (
                <PropertyCard key={property.id} property={property} />
              ))}
            </SimpleGrid>
          ) : (
            <Box textAlign="center" py={10}>
              <Text fontSize="xl">No properties match your criteria</Text>
              <Button mt={4} onClick={clearFilters}>Clear Filters</Button>
            </Box>
          )}
        </Container>
        
        <Footer />
      </Box>
    </>
  );
}

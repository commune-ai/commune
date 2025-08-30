 # start of file
import { useEffect, useState } from 'react';
import Head from 'next/head';
import { 
  Box, 
  Container, 
  Heading, 
  Text, 
  Button, 
  Flex, 
  Stack, 
  useColorModeValue, 
  SimpleGrid,
  Image
} from '@chakra-ui/react';
import { useWeb3 } from '../hooks/useWeb3';
import Header from '../components/Header';
import Footer from '../components/Footer';
import PropertyCard from '../components/PropertyCard';
import HeroSection from '../components/HeroSection';
import FeatureSection from '../components/FeatureSection';

export default function Home() {
  const { account, connect, isConnected } = useWeb3();
  const [properties, setProperties] = useState([]);
  
  useEffect(() => {
    // In a real app, we would fetch properties from the blockchain or API
    // For now, we'll use sample data
    setProperties([
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
    ]);
  }, []);
  
  return (
    <>
      <Head>
        <title>Home2Home | Rent-to-Own Real Estate Platform</title>
        <meta name="description" content="Revolutionizing rent-to-own through real estate tokenization" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      
      <Box minH="100vh" display="flex" flexDirection="column">
        <Header />
        
        <Box as="main" flex="1">
          <HeroSection />
          
          <Container maxW="container.xl" py={12}>
            <Heading as="h2" size="xl" textAlign="center" mb={12}>
              Available Properties
            </Heading>
            
            <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={10}>
              {properties.map((property) => (
                <PropertyCard key={property.id} property={property} />
              ))}
            </SimpleGrid>
            
            <Flex justify="center" mt={10}>
              <Button 
                size="lg" 
                colorScheme="blue" 
                as="a" 
                href="/properties"
              >
                View All Properties
              </Button>
            </Flex>
          </Container>
          
          <FeatureSection />
          
          <Box bg={useColorModeValue('gray.50', 'gray.900')} py={16}>
            <Container maxW="container.xl">
              <Heading as="h2" size="xl" textAlign="center" mb={8}>
                How It Works
              </Heading>
              
              <SimpleGrid columns={{ base: 1, md: 3 }} spacing={10} mt={10}>
                <Box textAlign="center" p={5}>
                  <Box 
                    rounded="full" 
                    bg="blue.500" 
                    color="white" 
                    w={12} 
                    h={12} 
                    display="flex" 
                    alignItems="center" 
                    justifyContent="center"
                    mx="auto"
                    mb={4}
                  >
                    1
                  </Box>
                  <Heading as="h3" size="md" mb={3}>Browse Properties</Heading>
                  <Text>Explore our selection of available homes and find the perfect match for your needs.</Text>
                </Box>
                
                <Box textAlign="center" p={5}>
                  <Box 
                    rounded="full" 
                    bg="blue.500" 
                    color="white" 
                    w={12} 
                    h={12} 
                    display="flex" 
                    alignItems="center" 
                    justifyContent="center"
                    mx="auto"
                    mb={4}
                  >
                    2
                  </Box>
                  <Heading as="h3" size="md" mb={3}>Sign Agreement</Heading>
                  <Text>Complete our streamlined application process and sign your rent-to-own agreement.</Text>
                </Box>
                
                <Box textAlign="center" p={5}>
                  <Box 
                    rounded="full" 
                    bg="blue.500" 
                    color="white" 
                    w={12} 
                    h={12} 
                    display="flex" 
                    alignItems="center" 
                    justifyContent="center"
                    mx="auto"
                    mb={4}
                  >
                    3
                  </Box>
                  <Heading as="h3" size="md" mb={3}>Build Equity</Heading>
                  <Text>Watch your ownership grow with each payment as you build equity in your future home.</Text>
                </Box>
              </SimpleGrid>
              
              <Flex justify="center" mt={12}>
                <Button 
                  size="lg" 
                  colorScheme="blue" 
                  as="a" 
                  href="/how-it-works"
                >
                  Learn More
                </Button>
              </Flex>
            </Container>
          </Box>
        </Box>
        
        <Footer />
      </Box>
    </>
  );
}

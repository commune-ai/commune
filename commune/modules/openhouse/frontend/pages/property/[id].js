 # start of file
import { useEffect, useState } from 'react';
import { useRouter } from 'next/router';
import Head from 'next/head';
import { 
  Box, 
  Container, 
  Heading, 
  Text, 
  Image, 
  Grid, 
  GridItem, 
  Flex, 
  Badge, 
  Button, 
  Stat, 
  StatLabel, 
  StatNumber, 
  StatHelpText, 
  Tabs, 
  TabList, 
  Tab, 
  TabPanels, 
  TabPanel,
  Progress,
  useToast
} from '@chakra-ui/react';
import Header from '../../components/Header';
import Footer from '../../components/Footer';
import { useWeb3 } from '../../hooks/useWeb3';

export default function PropertyDetail() {
  const router = useRouter();
  const { id } = router.query;
  const [property, setProperty] = useState(null);
  const [loading, setLoading] = useState(true);
  const { isConnected, connect, account } = useWeb3();
  const toast = useToast();
  
  useEffect(() => {
    if (!id) return;
    
    // In a real app, we would fetch property details from the blockchain or API
    // For now, we'll use sample data
    const fetchProperty = async () => {
      setLoading(true);
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Sample property data
      const sampleProperty = {
        id: parseInt(id),
        address: '123 Main St, Anytown, USA',
        price: 350000,
        bedrooms: 3,
        bathrooms: 2,
        squareFeet: 2000,
        yearBuilt: 2010,
        propertyType: 'Single Family',
        description: 'This beautiful home features an open floor plan, updated kitchen with stainless steel appliances, and a spacious backyard perfect for entertaining. The master bedroom includes an en-suite bathroom and walk-in closet.',
        images: ['/images/house1.jpg', '/images/house1-interior1.jpg', '/images/house1-interior2.jpg'],
        tokenAddress: '0x123456789abcdef',
        monthlyPayment: 2200,
        equityPercentage: 20, // 20% of payment goes to equity
        maintenancePercentage: 10, // 10% goes to maintenance fund
        ownershipProgress: 15, // Current tenant owns 15%
      };
      
      setProperty(sampleProperty);
      setLoading(false);
    };
    
    fetchProperty();
  }, [id]);
  
  const handleApply = async () => {
    if (!isConnected) {
      await connect();
      return;
    }
    
    toast({
      title: 'Application Started',
      description: 'Your application for this property has been initiated.',
      status: 'success',
      duration: 5000,
      isClosable: true,
    });
    
    // In a real app, we would redirect to an application form
    // or initiate a smart contract transaction
    router.push(`/application?propertyId=${id}`);
  };
  
  if (loading || !property) {
    return (
      <Box minH="100vh" display="flex" flexDirection="column">
        <Header />
        <Container maxW="container.xl" py={8} flex="1" textAlign="center">
          <Text>Loading property details...</Text>
        </Container>
        <Footer />
      </Box>
    );
  }
  
  return (
    <>
      <Head>
        <title>{property.address} | Home2Home</title>
        <meta name="description" content={`View details for ${property.address}`} />
      </Head>
      
      <Box minH="100vh" display="flex" flexDirection="column">
        <Header />
        
        <Container maxW="container.xl" py={8} flex="1">
          <Grid templateColumns={{ base: '1fr', lg: 'repeat(3, 1fr)' }} gap={8}>
            <GridItem colSpan={{ base: 1, lg: 2 }}>
              <Image 
                src={property.images[0]} 
                alt={property.address}
                borderRadius="lg"
                objectFit="cover"
                w="100%"
                h="400px"
              />
              
              <Flex mt={4} gap={4} overflowX="auto" pb={2}>
                {property.images.slice(1).map((img, idx) => (
                  <Image 
                    key={idx}
                    src={img} 
                    alt={`${property.address} - image ${idx + 2}`}
                    borderRadius="md"
                    objectFit="cover"
                    w="200px"
                    h="150px"
                    flexShrink={0}
                  />
                ))}
              </Flex>
              
              <Box mt={8}>
                <Heading as="h2" size="lg" mb={4}>About This Property</Heading>
                <Text fontSize="lg">{property.description}</Text>
              </Box>
              
              <Box mt={8}>
                <Heading as="h3" size="md" mb={4}>Property Details</Heading>
                <Grid templateColumns={{ base: '1fr', md: 'repeat(2, 1fr)' }} gap={4}>
                  <Stat>
                    <StatLabel>Property Type</StatLabel>
                    <StatNumber>{property.propertyType}</StatNumber>
                  </Stat>
                  <Stat>
                    <StatLabel>Year Built</StatLabel>
                    <StatNumber>{property.yearBuilt}</StatNumber>
                  </Stat>
                  <Stat>
                    <StatLabel>Square Feet</StatLabel>
                    <StatNumber>{property.squareFeet.toLocaleString()}</StatNumber>
                  </Stat>
                  <Stat>
                    <StatLabel>Bedrooms</StatLabel>
                    <StatNumber>{property.bedrooms}</StatNumber>
                  </Stat>
                  <Stat>
                    <StatLabel>Bathrooms</StatLabel>
                    <StatNumber>{property.bathrooms}</StatNumber>
                  </Stat>
                </Grid>
              </Box>
              
              <Box mt={8}>
                <Tabs variant="enclosed">
                  <TabList>
                    <Tab>Ownership Details</Tab>
                    <Tab>Payment Breakdown</Tab>
                    <Tab>Property History</Tab>
                  </TabList>
                  
                  <TabPanels>
                    <TabPanel>
                      <Box mb={4}>
                        <Text fontWeight="bold" mb={2}>Current Ownership</Text>
                        <Progress value={property.ownershipProgress} colorScheme="blue" height="24px" borderRadius="md" mb={2} />
                        <Flex justify="space-between">
                          <Text>{property.ownershipProgress}% Tenant</Text>
                          <Text>{100 - property.ownershipProgress}% Owner</Text>
                        </Flex>
                      </Box>
                      
                      <Text mb={4}>
                        With each monthly payment, you'll build equity in this property. 
                        {property.equityPercentage}% of your monthly payment will be converted to ownership tokens.
                      </Text>
                      
                      <Text fontWeight="bold">Token Address:</Text>
                      <Text fontFamily="monospace" mb={4}>{property.tokenAddress}</Text>
                    </TabPanel>
                    
                    <TabPanel>
                      <Stat mb={4}>
                        <StatLabel>Monthly Payment</StatLabel>
                        <StatNumber>${property.monthlyPayment.toLocaleString()}</StatNumber>
                      </Stat>
                      
                      <Box mb={4}>
                        <Text fontWeight="bold" mb={2}>Payment Allocation</Text>
                        <Grid templateColumns="1fr auto" gap={2}>
                          <Text>Rent Portion ({100 - property.equityPercentage - property.maintenancePercentage}%)</Text>
                          <Text>${Math.round(property.monthlyPayment * (100 - property.equityPercentage - property.maintenancePercentage) / 100).toLocaleString()}</Text>
                          
                          <Text>Equity Portion ({property.equityPercentage}%)</Text>
                          <Text>${Math.round(property.monthlyPayment * property.equityPercentage / 100).toLocaleString()}</Text>
                          
                          <Text>Maintenance Fund ({property.maintenancePercentage}%)</Text>
                          <Text>${Math.round(property.monthlyPayment * property.maintenancePercentage / 100).toLocaleString()}</Text>
                        </Grid>
                      </Box>
                    </TabPanel>
                    
                    <TabPanel>
                      <Text>Property transaction and maintenance history will be displayed here, tracked on the blockchain for complete transparency.</Text>
                    </TabPanel>
                  </TabPanels>
                </Tabs>
              </Box>
            </GridItem>
            
            <GridItem>
              <Box position="sticky" top="20px" p={6} borderWidth="1px" borderRadius="lg" bg="white">
                <Heading as="h1" size="lg" mb={2}>{property.address}</Heading>
                
                <Flex mb={4}>
                  <Badge colorScheme="blue" mr={2}>
                    {property.bedrooms} beds
                  </Badge>
                  <Badge colorScheme="green" mr={2}>
                    {property.bathrooms} baths
                  </Badge>
                  <Badge colorScheme="purple">
                    {property.squareFeet.toLocaleString()} sqft
                  </Badge>
                </Flex>
                
                <Stat mb={6}>
                  <StatLabel>Property Value</StatLabel>
                  <StatNumber>${property.price.toLocaleString()}</StatNumber>
                </Stat>
                
                <Stat mb={6}>
                  <StatLabel>Monthly Payment</StatLabel>
                  <StatNumber>${property.monthlyPayment.toLocaleString()}</StatNumber>
                  <StatHelpText>Includes equity building portion</StatHelpText>
                </Stat>
                
                <Button 
                  colorScheme="blue" 
                  size="lg" 
                  w="100%" 
                  mb={4}
                  onClick={handleApply}
                >
                  {isConnected ? 'Apply Now' : 'Connect Wallet to Apply'}
                </Button>
                
                <Text fontSize="sm" color="gray.600">
                  By applying, you'll start the process of entering a rent-to-own agreement for this property.
                </Text>
              </Box>
            </GridItem>
          </Grid>
        </Container>
        
        <Footer />
      </Box>
    </>
  );
}

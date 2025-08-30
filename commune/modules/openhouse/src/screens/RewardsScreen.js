import React, { useState } from 'react';
import { View, Text, StyleSheet, FlatList, TouchableOpacity, Modal, TextInput } from 'react-native';
import { useSelector, useDispatch } from 'react-redux';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { useTheme } from '../context/ThemeContext';
import { claimReward } from '../redux/actions/userActions';
import { addReward } from '../redux/actions/rewardActions';

const RewardsScreen = () => {
  const { theme } = useTheme();
  const dispatch = useDispatch();
  const { rewards, claimedRewards } = useSelector(state => state.rewards);
  const { stats } = useSelector(state => state.user);
  
  const [modalVisible, setModalVisible] = useState(false);
  const [newReward, setNewReward] = useState({
    title: '',
    description: '',
    cost: '',
    icon: 'gift',
    category: 'custom',
  });
  
  const handleClaimReward = (reward) => {
    const success = dispatch(claimReward(reward));
    if (!success) {
      // Show not enough points message
      alert('Not enough points to claim this reward!');
    }
  };
  
  const handleAddReward = () => {
    if (!newReward.title || !newReward.cost) {
      alert('Please enter a title and cost for your reward');
      return;
    }
    
    dispatch(addReward({
      ...newReward,
      cost: parseInt(newReward.cost, 10) || 100,
    }));
    
    setNewReward({
      title: '',
      description: '',
      cost: '',
      icon: 'gift',
      category: 'custom',
    });
    
    setModalVisible(false);
  };
  
  const renderRewardItem = ({ item }) => {
    const canAfford = stats.points >= item.cost;
    
    return (
      <View style={[styles.rewardCard, { backgroundColor: theme.secondaryBackground }]}>
        <View style={styles.rewardHeader}>
          <View style={[styles.iconContainer, { backgroundColor: theme.primary }]}>
            <Icon name={item.icon} size={24} color="#fff" />
          </View>
          <View style={styles.rewardInfo}>
            <Text style={[styles.rewardTitle, { color: theme.text }]}>{item.title}</Text>
            <Text style={[styles.rewardDescription, { color: theme.text }]}>{item.description}</Text>
          </View>
        </View>
        
        <View style={styles.rewardFooter}>
          <View style={styles.costContainer}>
            <Icon name="coin" size={16} color={theme.accent} />
            <Text style={[styles.costText, { color: theme.text }]}>{item.cost}</Text>
          </View>
          
          <TouchableOpacity 
            style={[styles.claimButton, { 
              backgroundColor: canAfford ? theme.primary : theme.border,
              opacity: canAfford ? 1 : 0.5,
            }]}
            onPress={() => handleClaimReward(item)}
            disabled={!canAfford}
          >
            <Text style={styles.claimButtonText}>Claim</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  };
  
  const renderClaimedItem = ({ item }) => {
    return (
      <View style={[styles.claimedCard, { backgroundColor: theme.secondaryBackground }]}>
        <View style={styles.rewardHeader}>
          <View style={[styles.iconContainer, { backgroundColor: theme.success }]}>
            <Icon name={item.icon} size={24} color="#fff" />
          </View>
          <View style={styles.rewardInfo}>
            <Text style={[styles.rewardTitle, { color: theme.text }]}>{item.title}</Text>
            <Text style={[styles.claimedDate, { color: theme.text }]}>
              Claimed on {new Date(item.claimedAt).toLocaleDateString()}
            </Text>
          </View>
        </View>
      </View>
    );
  };
  
  return (
    <View style={[styles.container, { backgroundColor: theme.background }]}>
      {/* Points Banner */}
      <View style={[styles.pointsBanner, { backgroundColor: theme.accent }]}>
        <Icon name="coin" size={24} color="#fff" />
        <Text style={styles.pointsText}>{stats.points} Points Available</Text>
      </View>
      
      {/* Available Rewards */}
      <Text style={[styles.sectionTitle, { color: theme.text }]}>Available Rewards</Text>
      <FlatList
        data={rewards}
        keyExtractor={item => item.id}
        renderItem={renderRewardItem}
        contentContainerStyle={styles.listContent}
      />
      
      {/* Recently Claimed */}
      {claimedRewards.length > 0 && (
        <>
          <Text style={[styles.sectionTitle, { color: theme.text }]}>Recently Claimed</Text>
          <FlatList
            data={claimedRewards.slice(0, 3)}
            keyExtractor={(item, index) => `${item.id}-${index}`}
            renderItem={renderClaimedItem}
            contentContainerStyle={styles.listContent}
          />
        </>
      )}
      
      {/* Add Custom Reward Button */}
      <TouchableOpacity 
        style={[styles.addButton, { backgroundColor: theme.primary }]}
        onPress={() => setModalVisible(true)}
      >
        <Icon name="plus" size={24} color="#fff" />
      </TouchableOpacity>
      
      {/* Add Reward Modal */}
      <Modal
        animationType="slide"
        transparent={true}
        visible={modalVisible}
        onRequestClose={() => setModalVisible(false)}
      >
        <View style={styles.modalContainer}>
          <View style={[styles.modalContent, { backgroundColor: theme.background }]}>
            <Text style={[styles.modalTitle, { color: theme.text }]}>Create Custom Reward</Text>
            
            <TextInput
              style={[styles.input, { backgroundColor: theme.secondaryBackground, color: theme.text }]}
              placeholder="Reward Title"
              placeholderTextColor="#888"
              value={newReward.title}
              onChangeText={(text) => setNewReward({...newReward, title: text})}
            />
            
            <TextInput
              style={[styles.input, { backgroundColor: theme.secondaryBackground, color: theme.text }]}
              placeholder="Description (optional)"
              placeholderTextColor="#888"
              value={newReward.description}
              onChangeText={(text) => setNewReward({...newReward, description: text})}
            />
            
            <TextInput
              style={[styles.input, { backgroundColor: theme.secondaryBackground, color: theme.text }]}
              placeholder="Cost in Points"
              placeholderTextColor="#888"
              keyboardType="numeric"
              value={newReward.cost}
              onChangeText={(text) => setNewReward({...newReward, cost: text})}
            />
            
            <View style={styles.modalButtons}>
              <TouchableOpacity 
                style={[styles.modalButton, { backgroundColor: theme.border }]}
                onPress={() => setModalVisible(false)}
              >
                <Text style={{ color: theme.text }}>Cancel</Text>
              </TouchableOpacity>
              
              <TouchableOpacity 
                style={[styles.modalButton, { backgroundColor: theme.primary }]}
                onPress={handleAddReward}
              >
                <Text style={{ color: '#fff' }}>Create</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
  },
  pointsBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
    borderRadius: 12,
    marginBottom: 20,
  },
  pointsText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
    marginLeft: 8,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 12,
    marginTop: 16,
  },
  listContent: {
    paddingBottom: 16,
  },
  rewardCard: {
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 1.5,
  },
  rewardHeader: {
    flexDirection: 'row',
    marginBottom: 12,
  },
  iconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
  },
  rewardInfo: {
    marginLeft: 12,
    flex: 1,
  },
  rewardTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  rewardDescription: {
    fontSize: 14,
    opacity: 0.7,
  },
  rewardFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  costContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  costText: {
    marginLeft: 4,
    fontWeight: 'bold',
  },
  claimButton: {
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 16,
  },
  claimButtonText: {
    color: '#fff',
    fontWeight: 'bold',
  },
  claimedCard: {
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    opacity: 0.8,
  },
  claimedDate: {
    fontSize: 12,
    opacity: 0.7,
  },
  addButton: {
    position: 'absolute',
    bottom: 24,
    right: 24,
    width: 56,
    height: 56,
    borderRadius: 28,
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 3,
  },
  modalContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.5)',
  },
  modalContent: {
    width: '80%',
    borderRadius: 12,
    padding: 20,
    elevation: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 16,
    textAlign: 'center',
  },
  input: {
    borderRadius: 8,
    padding: 12,
    marginBottom: 16,
  },
  modalButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  modalButton: {
    flex: 1,
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
    marginHorizontal: 4,
  },
});

export default RewardsScreen;

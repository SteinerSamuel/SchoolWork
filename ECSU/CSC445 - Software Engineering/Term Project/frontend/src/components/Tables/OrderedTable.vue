<template>
  <div>
    <md-table v-model=BookData :table-header-color="tableHeaderColor">
      <md-table-row slot="md-table-row" slot-scope="{item}" :key=componentKey>
        <md-table-cell md-label="Title">{{ item.Title }}</md-table-cell>
        <md-table-cell md-label="Author">{{ item.Author }}</md-table-cell>
        <md-table-cell md-label="Due Date">{{item.DueDate}}</md-table-cell>
        <md-table-cell md-label="Return"><md-button class="md-dense md-primary md-flat" v-on:click="return_book(item.Inventory_Id)">return</md-button></md-table-cell>
      </md-table-row>
    </md-table>
  </div>
</template>

<script>
import axios from "axios";
export default {
  name: "ordered-table",
  props: {
    tableHeaderColor: {
      type: String,
      default: ""
    },
    tableData: {
      type: String,
      default: "http://127.0.0.1:5000/rented/?user=%"
    }
  },
  methods: {
    return_book :function(inventory_id){
      axios.get("http://127.0.0.1:5000/return/?id=" + inventory_id);
      this.$emit('returned_book');
    }
  },
  data() {
    return {
      BookData: null,
      BookInfoData: "Test",
      showDialog: false,
      componentKey: 0
    };
  },
  mounted() {
    axios
      .get(this.tableData)
      .then(response => (this.BookData = response.data));
  }
};
</script>
